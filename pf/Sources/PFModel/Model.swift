import Foundation
import MLX
import MLXFast

// PFModel — the parity-proven privacy-filter forward, extracted verbatim from
// `pf-parity/main.swift` (Milestone 2) into a reusable library target so both the
// `pf-parity` parity check and the `pf` CLI (Phase C) share one model implementation.
//
// The numerical logic matches the oracle (cosine 1.0, argmax 20/20 vs
// apple/pf/parity-fixture.json, exported from ../../pf_mlx.py). fp32, windowed
// (banded) attention — O(n·window) memory, bit-identical to dense under the SWA
// mask (ported from pf_mlx._attn, D1) — sorted gatherMM MoE (D2), 33-label head.
// Codepoint-agnostic: `logits(_:)` takes token ids and returns [n, nCls].
//
// MLX targets must build via run.sh (xcodebuild) — `swift build` can't compile Metal.

// ---- config -------------------------------------------------------------------
public struct HP {
    public let nLayer, nHead, nHeadKV, headDim, nInter, nExpert, nExpertUsed, swaRadius, nCtxOrig: Int
    public let rmsEps: Float
    public let ropeTheta, yarnFactor, betaFast, betaSlow: Double
    public let truncate: Bool
    public var labels: [String]
}

func num(_ a: Any?) -> Double { (a as? NSNumber)?.doubleValue ?? 0 }
func int(_ a: Any?) -> Int { (a as? NSNumber)?.intValue ?? 0 }

func loadHP(_ dir: URL) throws -> HP {
    let cfg = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as! [String: Any]
    let rope = (cfg["rope_parameters"] as? [String: Any]) ?? (cfg["rope_scaling"] as? [String: Any]) ?? [:]
    let id2 = cfg["id2label"] as! [String: String]
    var labels = [String](repeating: "", count: id2.count)
    for (k, v) in id2 { labels[Int(k)!] = v }
    return HP(nLayer: int(cfg["num_hidden_layers"]), nHead: int(cfg["num_attention_heads"]),
              nHeadKV: int(cfg["num_key_value_heads"]), headDim: int(cfg["head_dim"]),
              nInter: int(cfg["intermediate_size"]), nExpert: int(cfg["num_local_experts"]),
              nExpertUsed: int(cfg["num_experts_per_tok"]), swaRadius: int(cfg["sliding_window"]),
              nCtxOrig: int(rope["original_max_position_embeddings"]),
              rmsEps: Float(num(cfg["rms_norm_eps"])), ropeTheta: num(rope["rope_theta"]),
              yarnFactor: num(rope["factor"]), betaFast: num(rope["beta_fast"]),
              betaSlow: num(rope["beta_slow"]), truncate: (rope["truncate"] as? Bool) ?? false,
              labels: labels)
}

// inv_freq + attn_factor — mirrors pf_mlx.yarn_inv_freq (Double precision).
func yarnInvFreq(_ hp: HP) -> ([Float], Float) {
    let half = hp.headDim / 2, base = hp.ropeTheta, factor = hp.yarnFactor
    func extrap(_ j: Int) -> Double { pow(base, -2.0 * Double(j) / Double(hp.headDim)) }
    if factor <= 1.0 { return ((0..<half).map { Float(extrap($0)) }, 1.0) }
    func corr(_ b: Double) -> Double { Double(hp.headDim) * log(Double(hp.nCtxOrig) / (b * 2 * .pi)) / (2 * log(base)) }
    var low = corr(hp.betaFast), high = corr(hp.betaSlow)
    if hp.truncate { low = low.rounded(.down); high = high.rounded(.up) }
    low = max(low, 0); high = min(high, Double(hp.headDim) - 1)
    let inv = (0..<half).map { j -> Float in
        let ramp = min(max((Double(j) - low) / max(high - low, 1e-3), 0), 1)
        return Float((extrap(j) / factor) * ramp + extrap(j) * (1 - ramp))
    }
    return (inv, Float(0.1 * log(factor) + 1.0))
}

// ---- weights ------------------------------------------------------------------
// An expert weight is either dense fp32, or an mlx `quantized(...)` triple
// (wq, scales, biases). `switchExpert` dispatches on this (pf_mlx `_switch`).
enum ExpertWeight {
    case dense(MLXArray)                                       // [E, out, in] (fp32 or fp16)
    case quant(wq: MLXArray, scales: MLXArray, biases: MLXArray?)  // mlx.quantize output
}

// Token embedding table: dense [V, D], or an 8-bit `quantized(...)` triple whose
// gathered rows are dequantized on lookup (pf_mlx `__call__` quantized-embedding branch).
enum Embedding {
    case dense(MLXArray)                                       // [V, D] (fp32 or fp16)
    case quant(wq: MLXArray, scales: MLXArray, biases: MLXArray?, bits: Int)

    // Embedding dimension D (rows = vocab). For the quant case D is recovered from the
    // dequantized group layout: cols(wq) packs `bits`-wide elements, scales has V×(D/group).
    var dim: Int {
        switch self {
        case .dense(let w): return w.shape[1]
        case .quant(let wq, _, _, let bits): return wq.shape[1] * (32 / bits)
        }
    }
    // Rows = vocab size (for the id clamp in `logits`).
    var rows: Int {
        switch self {
        case .dense(let w): return w.shape[0]
        case .quant(let wq, _, _, _): return wq.shape[0]
        }
    }
    // Gather rows for `ids` → [n, D]. Dense indexes directly; quant gathers the packed
    // rows + their scales/biases and dequantizes them to `dtype` (must match the dense
    // weights so the downstream matmuls agree — pf_mlx dequantizes into self.dtype).
    func gather(_ ids: MLXArray, qgroup: Int, dtype: DType) -> MLXArray {
        switch self {
        case .dense(let w):
            return w[ids]
        case .quant(let wq, let scales, let biases, let bits):
            return dequantized(wq[ids], scales: scales[ids], biases: biases?[ids],
                               groupSize: qgroup, bits: bits, dtype: dtype)
        }
    }
}

func quantEmbedding(_ w: MLXArray, qembed: Int, qgroup: Int) -> Embedding {
    guard qembed > 0 else { return .dense(w) }
    let (wq, scales, biases) = quantized(w, groupSize: qgroup, bits: qembed)  // `w` already cast to dtype
    return .quant(wq: wq, scales: scales, biases: biases, bits: qembed)
}

struct Layer {
    let attnNorm, wq, bq, wk, bk, wv, bv, wo, bo, sinks, postNorm, routerW, routerB: MLXArray
    let gateB, upB, downB: MLXArray
    let gateW, upW, downW: ExpertWeight
}

// Quantize an expert weight when qbits > 0, else keep it dense (qbits==0 path is
// byte-identical to the unquantized model). Mirrors pf_mlx.load_weights_mx `mq`.
func quantExpert(_ w: MLXArray, qbits: Int, qgroup: Int) -> ExpertWeight {
    guard qbits > 0 else { return .dense(w) }
    let (wq, scales, biases) = quantized(w, groupSize: qgroup, bits: qbits)
    return .quant(wq: wq, scales: scales, biases: biases)
}

// A PRE-quantized artifact (built by reference/export_mlx_quant.py) carries a
// `quantization` block in config.json; a bf16 checkpoint does not. Returns the block's
// {bits, group_size, embed_bits}, or nil for a bf16 checkpoint that must quantize at load.
func quantConfig(_ dir: URL) throws -> (bits: Int, group: Int, embedBits: Int)? {
    let cfg = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as! [String: Any]
    guard let q = cfg["quantization"] as? [String: Any] else { return nil }
    return (int(q["bits"]), int(q["group_size"]), int(q["embed_bits"]))
}

func loadModel(_ dir: URL, _ hp: HP, qbits: Int, qgroup: Int, dtype: DType)
    throws -> (MLXArray, [Layer], MLXArray, MLXArray, MLXArray) {
    let raw = try loadArrays(url: dir.appendingPathComponent("model.safetensors"))
    // All dense weights cast to the working dtype (fp32 for the parity reference, fp16 for
    // the shipped quantized config — mirrors pf_mlx.load_weights_mx's `c = astype(dtype)`).
    func f(_ k: String) -> MLXArray { raw[k]!.asType(dtype) }
    let I = hp.nInter
    var layers: [Layer] = []
    for i in 0..<hp.nLayer {
        let p = "model.layers.\(i)."
        let gateUp = raw[p + "mlp.experts.gate_up_proj"]!.swappedAxes(-1, -2).asType(dtype)  // [E,2I,D]
        let gub = raw[p + "mlp.experts.gate_up_proj_bias"]!.asType(dtype)                    // [E,2I]
        let downW = raw[p + "mlp.experts.down_proj"]!.swappedAxes(-1, -2).asType(dtype)       // [E,D,I]
        layers.append(Layer(
            attnNorm: f(p + "input_layernorm.weight"),
            wq: f(p + "self_attn.q_proj.weight"), bq: f(p + "self_attn.q_proj.bias"),
            wk: f(p + "self_attn.k_proj.weight"), bk: f(p + "self_attn.k_proj.bias"),
            wv: f(p + "self_attn.v_proj.weight"), bv: f(p + "self_attn.v_proj.bias"),
            wo: f(p + "self_attn.o_proj.weight"), bo: f(p + "self_attn.o_proj.bias"),
            sinks: f(p + "self_attn.sinks"), postNorm: f(p + "post_attention_layernorm.weight"),
            routerW: f(p + "mlp.router.weight"), routerB: f(p + "mlp.router.bias"),
            gateB: gub[0..., 0..<I], upB: gub[0..., I...],
            downB: f(p + "mlp.experts.down_proj_bias"),
            gateW: quantExpert(gateUp[0..., 0..<I, 0...], qbits: qbits, qgroup: qgroup),
            upW: quantExpert(gateUp[0..., I..., 0...], qbits: qbits, qgroup: qgroup),
            downW: quantExpert(downW, qbits: qbits, qgroup: qgroup)))
    }
    return (f("model.embed_tokens.weight"), layers, f("model.norm.weight"),
            f("score.weight"), f("score.bias"))
}

// Load a PRE-quantized artifact (reference/export_mlx_quant.py): the model's INTERNAL
// `w`-dict layout straight from safetensors — quantized experts/embedding as
// (wq,scales,biases) triples (`<key>.weight/.scales/.biases`), everything else dense fp16.
// No runtime quantize: the swapaxes/split/`mx.quantize` already happened at export, so this
// mirrors load_weights_mx's OUTPUT, not its transforms. Bit-identical to the runtime q-path
// that produced it (export's `--verify` proves cosine 1.0). The (wq,scales,biases) format is
// shared across MLX bindings, so Python-quantized tensors load straight into mlx-swift.
func loadModelPrequant(_ dir: URL, _ hp: HP, embedBits: Int, dtype: DType)
    throws -> (Embedding, [Layer], MLXArray, MLXArray, MLXArray) {
    let raw = try loadArrays(url: dir.appendingPathComponent("model.safetensors"))
    func d(_ k: String) -> MLXArray { raw[k]!.asType(dtype) }   // dense tensors -> working dtype
    func qexp(_ base: String) -> ExpertWeight {                 // (wq,scales,biases) triple, kept as-is
        .quant(wq: raw[base + ".weight"]!, scales: raw[base + ".scales"]!, biases: raw[base + ".biases"]!)
    }
    var layers: [Layer] = []
    for i in 0..<hp.nLayer {
        let o = "l\(i)."
        layers.append(Layer(
            attnNorm: d(o + "attn_norm"),
            wq: d(o + "wq"), bq: d(o + "bq"), wk: d(o + "wk"), bk: d(o + "bk"), wv: d(o + "wv"), bv: d(o + "bv"),
            wo: d(o + "wo"), bo: d(o + "bo"), sinks: d(o + "sinks"), postNorm: d(o + "post_norm"),
            routerW: d(o + "router_w"), routerB: d(o + "router_b"),
            gateB: d(o + "gate_b"), upB: d(o + "up_b"), downB: d(o + "down_b"),
            gateW: qexp(o + "gate_w"), upW: qexp(o + "up_w"), downW: qexp(o + "down_w")))
    }
    let embd = Embedding.quant(wq: raw["tok_embd.weight"]!, scales: raw["tok_embd.scales"]!,
                               biases: raw["tok_embd.biases"]!, bits: embedBits)
    return (embd, layers, d("output_norm"), d("cls_w"), d("cls_b"))
}

// ---- forward ------------------------------------------------------------------
func rope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray, _ n: Int, _ heads: Int, _ dh: Int) -> MLXArray {
    let half = dh / 2
    let xp = x.reshaped([n, heads, half, 2])
    let x0 = xp[.ellipsis, 0], x1 = xp[.ellipsis, 1]              // [n,heads,half]
    let c = cos.reshaped([n, 1, half]), s = sin.reshaped([n, 1, half])
    let o0 = x0 * c - x1 * s, o1 = x0 * s + x1 * c
    return stacked([o0, o1], axis: -1).reshaped([n, heads, dh])
}

func swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    let g = minimum(gate, 7.0), u = clip(up, min: -7.0, max: 7.0)
    return g * sigmoid(1.702 * g) * (u + 1.0)
}

// sorted expert matmul (pf_mlx _switch, sorted_indices=true path): rows are already
// expert-sorted so the gather runs one contiguous GEMM tile per expert. The explicit
// lhsIndices = arange(m) is REQUIRED on the sorted path — without it rows mispair.
// [m,1,in] -> [m,1,out]. Dense uses gatherMM; quantized uses gatherQuantizedMM
// (mlx `gather_qmm`, transpose=true so [E,out,in] stays in row-major weight layout).
func switchExpert(_ x: MLXArray, _ inds: MLXArray, _ weight: ExpertWeight,
                  _ bias: MLXArray, qbits: Int, qgroup: Int) -> MLXArray {
    let lhs = MLXArray(0..<x.shape[0])
    let y: MLXArray
    switch weight {
    case .dense(let w):
        y = gatherMM(x, w.swappedAxes(-1, -2), lhsIndices: lhs, rhsIndices: inds, sortedIndices: true)
    case .quant(let wq, let scales, let biases):
        y = gatherQuantizedMM(x, wq, scales: scales, biases: biases,
                              lhsIndices: lhs, rhsIndices: inds, transpose: true,
                              groupSize: qgroup, bits: qbits, sortedIndices: true)
    }
    return y + bias[inds].expandedDimensions(axis: -2)
}

func forward(_ ids: MLXArray, _ hp: HP, _ embd: Embedding, _ layers: [Layer],
             _ outNorm: MLXArray, _ clsW: MLXArray, _ clsB: MLXArray,
             _ inv: MLXArray, _ attnFactor: Float, qbits: Int, qgroup: Int, dtype: DType) -> MLXArray {
    let n = ids.shape[0], H = hp.nHead, Hkv = hp.nHeadKV, dh = hp.headDim
    let group = H / Hkv, scale = 1.0 / Float(Double(dh).squareRoot()), D = embd.dim
    // pos/theta in fp32 for precision (pf_mlx), then cos/sin cast to the working dtype so
    // they agree with the (dtype) weights in the attention matmuls.
    let pos = MLXArray(0..<n).asType(.float32).reshaped([n, 1])
    let theta = pos * inv.reshaped([1, dh / 2])                   // [n, half]
    let cos = (MLX.cos(theta) * attnFactor).asType(dtype), sin = (MLX.sin(theta) * attnFactor).asType(dtype)
    // Windowed (banded) attention geometry — mirrors pf_mlx._attn. Block size = R, so
    // each query block attends to its 3 neighbour blocks (<= 2R+1 keys). The additive
    // mask = band(|q-k|<=R) + validity(drop padding keys, 0<=kglob<n). Loop-invariant
    // (depends only on n,R), so build once: [nb,R,3R], O(n*window) memory not O(n^2).
    let R = hp.swaRadius
    let nb = (n + R - 1) / R
    let ii = MLXArray(0..<R).reshaped([R, 1])                     // query offset in block
    let jj = MLXArray(0..<(3 * R)).reshaped([1, 3 * R])           // key offset in 3-block window
    let band = (jj .>= ii) .&& (jj .<= ii + 2 * R)               // [R,3R] |q-k|<=R
    let kglob = (MLXArray(0..<nb).reshaped([nb, 1]) - 1) * R + MLXArray(0..<(3 * R)).reshaped([1, 3 * R])
    let valid = (kglob .>= 0) .&& (kglob .< n)                    // [nb,3R] drop padding keys
    let mask = MLX.where(band, 0.0, -1e9).asType(dtype).reshaped([1, R, 3 * R])
            + MLX.where(valid, 0.0, -1e9).asType(dtype).reshaped([nb, 1, 3 * R])  // [nb,R,3R]

    var x = embd.gather(ids, qgroup: qgroup, dtype: dtype)       // [n, D]
    for L in layers {
        // attention
        let h = MLXFast.rmsNorm(x, weight: L.attnNorm, eps: hp.rmsEps)
        let q = rope((matmul(h, L.wq.transposed()) + L.bq).reshaped([n, H, dh]), cos, sin, n, H, dh)
        let k0 = rope((matmul(h, L.wk.transposed()) + L.bk).reshaped([n, Hkv, dh]), cos, sin, n, Hkv, dh)
        let v0 = (matmul(h, L.wv.transposed()) + L.bv).reshaped([n, Hkv, dh])
        let k = repeated(k0, count: group, axis: 1), v = repeated(v0, count: group, axis: 1)  // GQA [n,H,dh]
        var qT = q.transposed(1, 0, 2), kT = k.transposed(1, 0, 2), vT = v.transposed(1, 0, 2) // [H,n,dh]
        // Windowed attention (pf_mlx._attn): pad seq to nb*R, then each query block
        // attends only to its 3 neighbour blocks (<=2R+1 keys) -> [H,nb,R,3R] scores.
        let pad = nb * R - n
        if pad > 0 {
            let zp = MLXArray.zeros([H, pad, dh], dtype: dtype)
            qT = concatenated([qT, zp], axis: 1); kT = concatenated([kT, zp], axis: 1); vT = concatenated([vT, zp], axis: 1)
        }
        let qb = qT.reshaped([H, nb, R, dh])                                   // query blocks
        let zb = MLXArray.zeros([H, 1, R, dh], dtype: dtype)
        let kp = concatenated([zb, kT.reshaped([H, nb, R, dh]), zb], axis: 1)  // pad 1 block each side
        let vp = concatenated([zb, vT.reshaped([H, nb, R, dh]), zb], axis: 1)
        let kn = concatenated([kp[0..., 0..<nb], kp[0..., 1..<(nb + 1)], kp[0..., 2..<(nb + 2)]], axis: 2) // [H,nb,3R,dh]
        let vn = concatenated([vp[0..., 0..<nb], vp[0..., 1..<(nb + 1)], vp[0..., 2..<(nb + 2)]], axis: 2)
        var scores = matmul(qb, kn.swappedAxes(-1, -2)) * scale                // [H,nb,R,3R]
        scores = scores + mask.reshaped([1, nb, R, 3 * R])                     // band + validity
        let sink = L.sinks.reshaped([H, 1, 1, 1])
        let m = maximum(scores.max(axis: -1, keepDims: true), sink)
        let e = exp(scores - m)
        let attn = e / (e.sum(axis: -1, keepDims: true) + exp(sink - m))
        let out = matmul(attn, vn).reshaped([H, nb * R, dh])[0..., 0..<n, 0...] // [H,n,dh] unpad
        let ao = out.transposed(1, 0, 2).reshaped([n, H * dh])                  // [n, H*dh]
        x = x + matmul(ao, L.wo.transposed()) + L.bo

        // MoE
        let hn = MLXFast.rmsNorm(x, weight: L.postNorm, eps: hp.rmsEps)
        let rl = matmul(hn, L.routerW.transposed()) + L.routerB                 // [n, E]
        let k_ = hp.nExpertUsed
        let inds = argPartition(-rl, kth: k_ - 1, axis: -1)[0..., 0..<k_]       // [n, k]
        let gw = softmax(takeAlong(rl, inds, axis: -1), axis: -1)               // [n, k]
        // Sort the n*k (token,slot) pairs by expert so each gatherMM is one contiguous
        // tile per expert, then unsort (pf_mlx._moe). Pure reordering -> numerically identical.
        let flat = inds.reshaped([-1])                                          // [n*k] expert per (token,slot)
        let order = argSort(flat)                                               // [n*k]
        let sinds = flat[order]                                                 // expert ids, sorted ascending
        let xe = repeated(hn, count: k_, axis: 0)[order].expandedDimensions(axis: -2)  // [n*k,1,D] rows in expert order
        let gate = switchExpert(xe, sinds, L.gateW, L.gateB, qbits: qbits, qgroup: qgroup)  // [n*k,1,I]
        let up = switchExpert(xe, sinds, L.upW, L.upB, qbits: qbits, qgroup: qgroup)
        let hh = swiglu(gate, up)
        let oeSorted = switchExpert(hh, sinds, L.downW, L.downB, qbits: qbits, qgroup: qgroup)  // [n*k,1,D]
        let oe = oeSorted.reshaped([n * k_, -1])[argSort(order)].reshaped([n, k_, D])  // unsort -> [n,k,D]
        x = x + (oe * gw.reshaped([n, k_, 1])).sum(axis: 1)                     // [n, D]
    }
    let xf = MLXFast.rmsNorm(x, weight: outNorm, eps: hp.rmsEps)
    return matmul(xf, clsW.transposed()) + clsB                                 // [n, n_cls]
}

// ---- public model -------------------------------------------------------------
/// Reusable privacy-filter model: loads weights once (fp32 unquantized, or 4-bit MoE +
/// 8-bit embed in fp16 — see `init`), then runs the parity-proven forward. Shared by
/// `pf-parity` (both configs) and `pf` (quantized default).
public struct Model {
    public let hp: HP
    private let outNorm, clsW, clsB, inv: MLXArray
    private let embd: Embedding
    private let layers: [Layer]
    private let attnFactor: Float
    private let qbits, qgroup: Int
    private let dtype: DType

    /// Load config (`config.json`) + weights (`model.safetensors`) from `modelDir`.
    ///
    /// - `qbits == 0`  → fp32, fully unquantized (byte-identical to the M2 parity setup;
    ///   cosine 1.0). Dense weights stay fp32.
    /// - `qbits > 0`   → quantize each MoE expert weight (gate/up/down) to `qbits` at `qgroup`,
    ///   and run in fp16 (dense weights cast to fp16 too — pf_mlx's "870 MB" config).
    /// - `qembed > 0`  → quantize the token-embedding table to `qembed` bits (else dense).
    ///
    /// Quantization is LOSSY: the quantized model targets cosine ≥ 0.995 / argmax ≥ 99%
    /// vs the fp32 fixtures (matches pf_mlx's measured 4-bit ≈ 0.998). The shipped CLI uses
    /// the 4/64/8 default (~870 MB); pass `qbits: 0` for the exact-parity reference path.
    ///
    /// If `modelDir` is a PRE-quantized artifact (config.json has a `quantization` block,
    /// produced by reference/export_mlx_quant.py) the triples are loaded straight in and the
    /// `qbits`/`qgroup`/`qembed` args are IGNORED — the on-disk block is authoritative.
    public init(modelDir: URL, qbits: Int = 4, qgroup: Int = 64, qembed: Int = 8) throws {
        let hp = try loadHP(modelDir)
        let embd: Embedding
        let layers: [Layer]
        let outNorm: MLXArray
        let clsW: MLXArray
        let clsB: MLXArray
        let effQbits: Int
        let effQgroup: Int
        let dtype: DType

        if let q = try quantConfig(modelDir) {
            // PRE-quantized artifact: load (wq,scales,biases) triples directly, no runtime
            // quantize. Always fp16 (matches export's working dtype). Init args ignored.
            dtype = .float16
            effQbits = q.bits
            effQgroup = q.group
            let (e, l, on, cw, cb) = try loadModelPrequant(modelDir, hp, embedBits: q.embedBits, dtype: dtype)
            embd = e; layers = l; outNorm = on; clsW = cw; clsB = cb
        } else {
            // bf16 checkpoint: fp32 for the exact parity reference; fp16 once we quantize.
            dtype = qbits > 0 ? .float16 : .float32
            effQbits = qbits
            effQgroup = qgroup
            let (embdW, l, on, cw, cb) = try loadModel(modelDir, hp, qbits: qbits, qgroup: qgroup, dtype: dtype)
            embd = quantEmbedding(embdW, qembed: qembed, qgroup: qgroup)
            layers = l; outNorm = on; clsW = cw; clsB = cb
        }

        let (invArr, attnFactor) = yarnInvFreq(hp)
        self.hp = hp
        self.embd = embd
        self.layers = layers
        self.outNorm = outNorm
        self.clsW = clsW
        self.clsB = clsB
        self.inv = MLXArray(invArr)
        self.attnFactor = attnFactor
        self.qbits = effQbits
        self.qgroup = effQgroup
        self.dtype = dtype
    }

    /// Per-token classification logits for `ids`. Returns an MLXArray of shape
    /// [n, nCls] (nCls == hp.labels.count). Pure move of the M2 forward.
    ///
    /// Token ids are clamped into `0..<vocab` before the embedding gather. An id outside
    /// that range triggers an MLX gather abort that a Swift `do/catch` CANNOT intercept,
    /// killing the whole stream. Clamping keeps every id in-range (over-redaction or a
    /// garbage label for a clamped id is acceptable; an uncatchable abort is not). In-range
    /// ids are untouched, so the parity math is unchanged.
    public func logits(_ ids: [Int]) -> MLXArray {
        let vocab = embd.rows              // embed_tokens rows
        let safe = ids.map { Swift.max(0, Swift.min($0, vocab - 1)) }
        let idsArr = MLXArray(safe.map { Int32($0) })
        return forward(idsArr, hp, embd, layers, outNorm, clsW, clsB, inv, attnFactor,
                       qbits: qbits, qgroup: qgroup, dtype: dtype)
    }

    /// Total bytes of every stored weight array — including the `(wq, scales, biases)`
    /// triples for quantized experts/embedding. Used by `pf-parity` to confirm the
    /// quantized footprint (~870 MB vs ~2.7 GB fp32). Evaluates the arrays once.
    public var weightBytes: Int {
        func dense(_ a: MLXArray) -> Int { a.nbytes }
        func expert(_ w: ExpertWeight) -> Int {
            switch w {
            case .dense(let a): return a.nbytes
            case .quant(let wq, let sc, let b): return wq.nbytes + sc.nbytes + (b?.nbytes ?? 0)
            }
        }
        var total = 0
        switch embd {
        case .dense(let a): total += a.nbytes
        case .quant(let wq, let sc, let b, _): total += wq.nbytes + sc.nbytes + (b?.nbytes ?? 0)
        }
        total += dense(outNorm) + dense(clsW) + dense(clsB) + dense(inv)
        for L in layers {
            total += [L.attnNorm, L.wq, L.bq, L.wk, L.bk, L.wv, L.bv, L.wo, L.bo,
                      L.sinks, L.postNorm, L.routerW, L.routerB,
                      L.gateB, L.upB, L.downB].reduce(0) { $0 + $1.nbytes }
            total += expert(L.gateW) + expert(L.upW) + expert(L.downW)
        }
        return total
    }
}
