#!/usr/bin/env python3
"""MLX (Metal GPU) port of the privacy-filter forward — the SPARSE path.

The ANE experiment showed dense-MoE is accurate but slow (the 128 experts fall
to the GPU and run 32x redundant). MLX runs the MoE *sparsely* on the Metal GPU
via mx.gather_mm (top-4 of 128, ~32x less compute), which is where the GGUF
engine's speed comes from. Runs on Apple Silicon only (Metal).

Self-contained: mlx + numpy + tokenizers (no torch). Math mirrors the SSOT
reference apple/pf_model.py; the HF->weights mapping mirrors scripts/convert.py.
The sparse MoE follows mlx-lm's SwitchLinear (gather_mm + bias gather).

Validates against the PC-dumped apple/ref.npz (fp32 reference logits):
  uv run --with mlx --with numpy --with tokenizers python apple/pf_mlx.py \
      apple/models/privacy-filter apple/ref.npz --dtype fp32
  # then fp16 for the speed number:
  ... --dtype fp16
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass
class HP:  # mirror src/gguf_loader.h / pf_model.HP
    n_layer: int; n_embd: int; n_head: int; n_head_kv: int; head_dim: int
    n_inter: int; n_expert: int; n_expert_used: int; swa_radius: int
    n_ctx_orig: int; rms_eps: float; rope_base: float
    yarn_factor: float; yarn_beta_fast: float; yarn_beta_slow: float; yarn_truncate: bool
    labels: list[str]

    @property
    def n_cls(self) -> int:
        return len(self.labels)


def load_hp(model_dir: Path) -> HP:
    cfg = json.loads((model_dir / "config.json").read_text())
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    id2 = {int(k): v for k, v in cfg["id2label"].items()}
    return HP(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"],
              cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"],
              cfg["num_local_experts"], cfg["num_experts_per_tok"], cfg["sliding_window"],
              rope["original_max_position_embeddings"], cfg["rms_norm_eps"], rope["rope_theta"],
              rope["factor"], rope["beta_fast"], rope["beta_slow"], bool(rope.get("truncate", False)),
              [id2[i] for i in range(len(id2))])


def yarn_inv_freq(hp: HP) -> tuple[np.ndarray, float]:
    """inv_freq[j] = base^(-2j/dh)/ff[j] and attn_factor — mirrors model.cpp."""
    half = hp.head_dim // 2
    base, factor = hp.rope_base, hp.yarn_factor
    j = np.arange(half, dtype=np.float64)
    extrap = base ** (-2.0 * j / hp.head_dim)
    if factor <= 1.0:
        return extrap.astype(np.float32), 1.0
    corr = lambda b: hp.head_dim * math.log(hp.n_ctx_orig / (b * 2 * math.pi)) / (2 * math.log(base))
    low, high = corr(hp.yarn_beta_fast), corr(hp.yarn_beta_slow)
    if hp.yarn_truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0.0), min(high, hp.head_dim - 1.0)
    ramp = np.clip((j - low) / max(high - low, 1e-3), 0.0, 1.0)
    inv_freq = (extrap / factor) * ramp + extrap * (1.0 - ramp)   # yarn-corrected freq
    return inv_freq.astype(np.float32), 0.1 * math.log(factor) + 1.0


def load_weights_mx(model_dir: Path, hp: HP, dtype: mx.Dtype) -> dict[str, mx.array]:
    """safetensors -> our layout (mx). Mapping mirrors scripts/convert.py."""
    raw: dict[str, mx.array] = {}
    for p in sorted(model_dir.glob("model*.safetensors")):
        raw.update(mx.load(str(p)))
    if not raw:
        sys.exit(f"no model*.safetensors in {model_dir}")
    inter = hp.n_inter
    c = lambda a: a.astype(dtype)
    qbits = int(os.environ.get("PF_QBITS", "4"))   # default 4-bit MoE (870MB config); PF_QBITS=0 -> fp16
    qgs = int(os.environ.get("PF_QGROUP", "64"))
    def mq(a):  # fp16 array, or a [w_q, scales, biases] list if quantizing
        return mx.quantize(c(a), group_size=qgs, bits=qbits) if qbits else c(a)
    qembed = int(os.environ.get("PF_QEMBED", "8"))   # default 8-bit token embeddings; PF_QEMBED=0 -> fp16
    def mqe(a):
        return mx.quantize(c(a), group_size=qgs, bits=qembed) if qembed else c(a)
    w = {"tok_embd": mqe(raw["model.embed_tokens.weight"]), "output_norm": c(raw["model.norm.weight"]),
         "cls_w": c(raw["score.weight"]), "cls_b": c(raw["score.bias"])}
    for i in range(hp.n_layer):
        p, o = f"model.layers.{i}.", f"l{i}."
        w[o + "attn_norm"] = c(raw[p + "input_layernorm.weight"])
        for q in ("q", "k", "v"):
            w[o + f"w{q}"] = c(raw[p + f"self_attn.{q}_proj.weight"])
            w[o + f"b{q}"] = c(raw[p + f"self_attn.{q}_proj.bias"])
        w[o + "wo"] = c(raw[p + "self_attn.o_proj.weight"])
        w[o + "bo"] = c(raw[p + "self_attn.o_proj.bias"])
        w[o + "sinks"] = c(raw[p + "self_attn.sinks"])
        w[o + "post_norm"] = c(raw[p + "post_attention_layernorm.weight"])
        w[o + "router_w"] = c(raw[p + "mlp.router.weight"])
        w[o + "router_b"] = c(raw[p + "mlp.router.bias"])
        gate_up = mx.swapaxes(raw[p + "mlp.experts.gate_up_proj"], -1, -2)   # [E, 2I, D]
        w[o + "gate_w"] = mq(gate_up[:, :inter, :])
        w[o + "up_w"] = mq(gate_up[:, inter:, :])
        gub = raw[p + "mlp.experts.gate_up_proj_bias"]
        w[o + "gate_b"] = c(gub[:, :inter])
        w[o + "up_b"] = c(gub[:, inter:])
        w[o + "down_w"] = mq(mx.swapaxes(raw[p + "mlp.experts.down_proj"], -1, -2))  # [E, D, I]
        w[o + "down_b"] = c(raw[p + "mlp.experts.down_proj_bias"])
    return w


def swiglu_oai(gate: mx.array, up: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return gate * mx.sigmoid(alpha * gate) * (up + 1.0)


class PFMLX:
    def __init__(self, w: dict[str, mx.array], hp: HP, dtype: mx.Dtype) -> None:
        self.w, self.hp, self.dtype = w, hp, dtype
        inv, self.attn_factor = yarn_inv_freq(hp)
        self.inv = mx.array(inv)
        self.qbits = int(os.environ.get("PF_QBITS", "4"))   # MoE quant (matches load_weights_mx)
        self.qgs = int(os.environ.get("PF_QGROUP", "64"))
        self.qembed = int(os.environ.get("PF_QEMBED", "8"))  # token-embedding quant

    def _rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x [n, heads, dh], interleaved (2i, 2i+1) pairs
        x0, x1 = x[..., 0::2], x[..., 1::2]
        c, s = cos[:, None, :], sin[:, None, :]
        o0, o1 = x0 * c - x1 * s, x0 * s + x1 * c
        return mx.stack([o0, o1], axis=-1).reshape(x.shape)

    def _switch(self, x: mx.array, inds: mx.array, weight, bias: mx.array) -> mx.array:
        # expert matmul on expert-sorted rows -> one contiguous GEMM tile per expert. [m,1,in]->[m,1,out]
        lhs = mx.arange(x.shape[0])
        if isinstance(weight, (tuple, list)):  # quantized [w_q, scales, biases] -> gather_qmm
            wq, sc, qb = weight
            y = mx.gather_qmm(x, wq, sc, qb, lhs_indices=lhs, rhs_indices=inds, transpose=True,
                              group_size=self.qgs, bits=self.qbits, sorted_indices=True)
        else:                          # fp16 -> gather_mm
            y = mx.gather_mm(x, mx.swapaxes(weight, -1, -2), lhs_indices=lhs, rhs_indices=inds, sorted_indices=True)
        return y + mx.expand_dims(bias[inds], -2)

    def _moe(self, x: mx.array, o: str) -> mx.array:
        hp, w = self.hp, self.w
        rl = x @ w[o + "router_w"].T + w[o + "router_b"]              # [n, E]
        k = hp.n_expert_used
        inds = mx.argpartition(-rl, kth=k - 1, axis=-1)[:, :k]        # [n, k] top-k
        gw = mx.softmax(mx.take_along_axis(rl, inds, axis=-1), axis=-1)   # softmax over top-k
        n = x.shape[0]
        # sort (token,slot) pairs by expert -> gather_mm runs one contiguous tile per expert.
        # Big win at long n (~256 tokens/expert at 8k => ~2.7x vs scattered gather); +14% at 512.
        flat = inds.reshape(-1)                                      # [n*k] expert per (token,slot)
        order = mx.argsort(flat)                                     # [n*k]
        sinds = flat[order]                                          # expert ids, sorted ascending
        xe = mx.expand_dims(mx.repeat(x, k, axis=0)[order], -2)      # [n*k,1,D] rows in expert order
        h = swiglu_oai(self._switch(xe, sinds, w[o + "gate_w"], w[o + "gate_b"]),
                       self._switch(xe, sinds, w[o + "up_w"], w[o + "up_b"]))   # [n*k,1,I]
        oe = self._switch(h, sinds, w[o + "down_w"], w[o + "down_b"])           # [n*k,1,D]
        oe = oe.reshape(n * k, -1)[mx.argsort(order)].reshape(n, k, -1)         # unsort -> [n,k,D]
        return (oe * gw[..., None]).sum(axis=-2)                     # [n, D]

    def __call__(self, ids: mx.array) -> mx.array:
        hp, w = self.hp, self.w
        n = ids.shape[0]
        pos = mx.arange(n).astype(mx.float32)
        theta = pos[:, None] * self.inv[None, :]
        cos = (mx.cos(theta) * self.attn_factor).astype(self.dtype)
        sin = (mx.sin(theta) * self.attn_factor).astype(self.dtype)

        te = w["tok_embd"]
        if isinstance(te, (tuple, list)):  # quantized embedding: dequantize the gathered rows
            twq, tsc, tqb = te
            x = mx.dequantize(twq[ids], tsc[ids], tqb[ids], group_size=self.qgs, bits=self.qembed)
        else:
            x = te[ids]                                              # [n, D]
        for i in range(hp.n_layer):
            o = f"l{i}."
            x = x + self._attn(self._norm(x, w[o + "attn_norm"]), o, cos, sin)
            x = x + self._moe(self._norm(x, w[o + "post_norm"]), o)
        x = self._norm(x, w["output_norm"])
        return x @ w["cls_w"].T + w["cls_b"]                         # [n, n_cls]

    def _attn(self, h: mx.array, o: str, cos: mx.array, sin: mx.array) -> mx.array:
        """Windowed (banded) attention + sinks: O(n*window) memory, not O(n^2).

        Exactly equals dense attention under the SWA mask (radius R) but never
        materialises the full n x n scores -> linear memory, no OOM wall at long n.
        Block size = R, so each query block attends to its 3 neighbour blocks
        (<= 2R+1 keys). Padding blocks/positions are masked out by `valid`; the
        attention sink keeps the denominator > 0 so masked rows stay finite.
        """
        hp, w = self.hp, self.w
        n = h.shape[0]
        H, Hkv, dh, R = hp.n_head, hp.n_head_kv, hp.head_dim, hp.swa_radius
        group, scale = H // Hkv, 1.0 / math.sqrt(dh)
        q = self._rope((h @ w[o + "wq"].T + w[o + "bq"]).reshape(n, H, dh), cos, sin)
        k = self._rope((h @ w[o + "wk"].T + w[o + "bk"]).reshape(n, Hkv, dh), cos, sin)
        v = (h @ w[o + "wv"].T + w[o + "bv"]).reshape(n, Hkv, dh)
        k = mx.repeat(k, group, axis=1); v = mx.repeat(v, group, axis=1)            # GQA -> [n,H,dh]
        q = mx.transpose(q, (1, 0, 2)); k = mx.transpose(k, (1, 0, 2)); v = mx.transpose(v, (1, 0, 2))
        nb = (n + R - 1) // R
        pad = nb * R - n
        if pad:
            zp = mx.zeros((H, pad, dh), dtype=self.dtype)
            q = mx.concatenate([q, zp], axis=1); k = mx.concatenate([k, zp], axis=1); v = mx.concatenate([v, zp], axis=1)
        qb = q.reshape(H, nb, R, dh)                                                # query blocks
        zb = mx.zeros((H, 1, R, dh), dtype=self.dtype)
        kp = mx.concatenate([zb, k.reshape(H, nb, R, dh), zb], axis=1)              # pad 1 block each side
        vp = mx.concatenate([zb, v.reshape(H, nb, R, dh), zb], axis=1)
        kn = mx.concatenate([kp[:, 0:nb], kp[:, 1:nb + 1], kp[:, 2:nb + 2]], axis=2)  # [H,nb,3R,dh] neighbours
        vn = mx.concatenate([vp[:, 0:nb], vp[:, 1:nb + 1], vp[:, 2:nb + 2]], axis=2)
        scores = (qb @ mx.swapaxes(kn, -1, -2)) * scale                            # [H,nb,R,3R]
        ii, jj = mx.arange(R)[:, None], mx.arange(3 * R)[None, :]
        band = mx.logical_and(jj >= ii, jj <= ii + 2 * R)                          # [R,3R] |q-k|<=R
        kglob = (mx.arange(nb)[:, None] - 1) * R + mx.arange(3 * R)[None, :]        # [nb,3R] key global idx
        valid = mx.logical_and(kglob >= 0, kglob < n)                              # drop padding keys
        mask = (mx.where(band, 0.0, -1e9).astype(self.dtype)[None, :, :]
                + mx.where(valid, 0.0, -1e9).astype(self.dtype)[:, None, :])       # [nb,R,3R]
        scores = scores + mask[None]
        sink = w[o + "sinks"].reshape(H, 1, 1, 1)
        m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
        e = mx.exp(scores - m)
        attn = e / (mx.sum(e, axis=-1, keepdims=True) + mx.exp(sink - m))
        out = (attn @ vn).reshape(H, nb * R, dh)[:, :n, :]                         # [H,n,dh] unpad
        ao = mx.transpose(out, (1, 0, 2)).reshape(n, H * dh)
        return ao @ w[o + "wo"].T + w[o + "bo"]

    def _norm(self, x: mx.array, weight: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight, self.hp.rms_eps)


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("ref", type=Path, help="ref.npz from dump_reference.py")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    materialize = mx.eval   # MLX force-evaluate (lazy graph -> compute)
    dtype = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[args.dtype]
    hp = load_hp(args.model_dir)
    model = PFMLX(load_weights_mx(args.model_dir, hp, dtype), hp, dtype)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    d = np.load(args.ref)
    texts, ref_log, lengths = d["texts"], d["ref_logits"], d["lengths"]

    print(f"MLX dtype={args.dtype}, {len(texts)} texts, {hp.n_cls} labels\n")
    hit = tot = 0
    coss = []
    last_ids = None
    for t, text in enumerate(texts):
        ids = mx.array(tok.encode(str(text)).ids, dtype=mx.int32)
        last_ids = ids
        out = np.array(model(ids).astype(mx.float32))               # [n, n_cls]
        n = int(lengths[t])
        a, b = out[:n], ref_log[t, :n]
        hit += int((a.argmax(-1) == b.argmax(-1)).sum()); tot += n
        coss.append(float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    print(f"MLX vs fp32 ref:  argmax-agree={hit / tot * 100:.1f}%   mean cosine={np.mean(coss):.5f}")

    ids = last_ids
    materialize(model(ids))                                          # warmup
    t0 = time.perf_counter()
    for _ in range(50):
        materialize(model(ids))
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"latency: {ms:.2f} ms/forward  ({int(ids.shape[0])} tokens, {args.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
