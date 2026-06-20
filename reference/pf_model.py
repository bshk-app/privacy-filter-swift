#!/usr/bin/env python3
"""Faithful PyTorch reference for the openai-privacy-filter forward pass.

Step 1 of the Apple ANE experiment. This reimplements the EXACT math of the
ggml graph in plain PyTorch (fp32, gather-MoE) so we have a correct, runnable
reference *before* reshaping anything for the Neural Engine (step 2). Once the
ANE-shaped Core ML variant exists, we validate it against THIS.

SSOT:
  - forward graph .......... ../src/model.cpp  (pf::model::forward)
  - hparams ................ ../src/gguf_loader.h (pf::hparams)
  - HF -> weights mapping .. ../scripts/convert.py  (keep in sync)

Run (after `bash apple/download.sh`):
  uv run python apple/pf_model.py apple/models/privacy-filter \
      "Contact John Doe at jdoe@example.com"
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# hyper-parameters — mirror src/gguf_loader.h, read straight from config.json
# --------------------------------------------------------------------------- #
@dataclass
class HP:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_inter: int          # expert FF dim (== n_embd for this model: square experts)
    n_expert: int
    n_expert_used: int
    swa_radius: int       # |q-k| <= swa_radius is visible (bidirectional)
    n_ctx_orig: int
    rms_eps: float
    rope_base: float
    yarn_factor: float
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_truncate: bool
    labels: list[str]

    @property
    def n_cls(self) -> int:
        return len(self.labels)


def load_hp(model_dir: Path) -> HP:
    cfg = json.loads((model_dir / "config.json").read_text())
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    return HP(
        n_layer=cfg["num_hidden_layers"],
        n_embd=cfg["hidden_size"],
        n_head=cfg["num_attention_heads"],
        n_head_kv=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        n_inter=cfg["intermediate_size"],
        n_expert=cfg["num_local_experts"],
        n_expert_used=cfg["num_experts_per_tok"],
        swa_radius=cfg["sliding_window"],
        n_ctx_orig=rope["original_max_position_embeddings"],
        rms_eps=cfg["rms_norm_eps"],
        rope_base=rope["rope_theta"],
        yarn_factor=rope["factor"],
        yarn_beta_fast=rope["beta_fast"],
        yarn_beta_slow=rope["beta_slow"],
        yarn_truncate=bool(rope.get("truncate", False)),
        labels=[id2label[i] for i in range(len(id2label))],
    )


# --------------------------------------------------------------------------- #
# YaRN frequency factors — line-for-line port of model.cpp::yarn_freq_factors
# --------------------------------------------------------------------------- #
def yarn_freq_factors(hp: HP) -> tuple[torch.Tensor, float]:
    half = hp.head_dim // 2
    base, factor = hp.rope_base, hp.yarn_factor
    ff = torch.ones(half, dtype=torch.float32)
    if factor <= 1.0:
        return ff, 1.0

    def corr_dim(beta: float) -> float:
        return hp.head_dim * math.log(hp.n_ctx_orig / (beta * 2.0 * math.pi)) / (2.0 * math.log(base))

    low, high = corr_dim(hp.yarn_beta_fast), corr_dim(hp.yarn_beta_slow)
    if hp.yarn_truncate:
        low, high = math.floor(low), math.ceil(high)
    low = max(low, 0.0)
    high = min(high, hp.head_dim - 1.0)
    for j in range(half):
        extrap = base ** (-2.0 * j / hp.head_dim)          # original inv_freq
        interp = extrap / factor
        ramp = min(max((j - low) / max(high - low, 1e-3), 0.0), 1.0)
        inv_freq = interp * ramp + extrap * (1.0 - ramp)
        ff[j] = extrap / inv_freq                           # ggml divides theta by ff
    return ff, 0.1 * math.log(factor) + 1.0                 # (ff, attn_factor / mscale)


# --------------------------------------------------------------------------- #
# weight loading — HF safetensors -> our flat dict. Mirrors scripts/convert.py.
# --------------------------------------------------------------------------- #
def load_weights(model_dir: Path, hp: HP) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    inter = hp.n_inter
    raw: dict[str, torch.Tensor] = {}
    for p in sorted(model_dir.glob("model*.safetensors")):
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    if not raw:
        sys.exit(f"no model*.safetensors found in {model_dir}")

    def g(name: str) -> torch.Tensor:
        return raw[name].float()

    w: dict[str, torch.Tensor] = {
        "tok_embd": g("model.embed_tokens.weight"),
        "output_norm": g("model.norm.weight"),
        "cls_w": g("score.weight"),
        "cls_b": g("score.bias"),
    }
    for i in range(hp.n_layer):
        p, o = f"model.layers.{i}.", f"l{i}."
        w[o + "attn_norm"] = g(p + "input_layernorm.weight")
        for proj in ("q", "k", "v"):
            w[o + f"w{proj}"] = g(p + f"self_attn.{proj}_proj.weight")
            w[o + f"b{proj}"] = g(p + f"self_attn.{proj}_proj.bias")
        w[o + "wo"] = g(p + "self_attn.o_proj.weight")
        w[o + "bo"] = g(p + "self_attn.o_proj.bias")
        w[o + "sinks"] = g(p + "self_attn.sinks")
        w[o + "post_norm"] = g(p + "post_attention_layernorm.weight")
        w[o + "router_w"] = g(p + "mlp.router.weight")
        w[o + "router_b"] = g(p + "mlp.router.bias")
        # experts.gate_up_proj: [E, in, 2*inter] -> [E, 2*inter, in], split halves
        gate_up = g(p + "mlp.experts.gate_up_proj").transpose(-1, -2)
        w[o + "gate_w"] = gate_up[:, :inter, :].contiguous()      # [E, inter, in]
        w[o + "up_w"] = gate_up[:, inter:, :].contiguous()
        gub = g(p + "mlp.experts.gate_up_proj_bias")
        w[o + "gate_b"] = gub[:, :inter].contiguous()
        w[o + "up_b"] = gub[:, inter:].contiguous()
        w[o + "down_w"] = g(p + "mlp.experts.down_proj").transpose(-1, -2).contiguous()
        w[o + "down_b"] = g(p + "mlp.experts.down_proj_bias")
    return w


# --------------------------------------------------------------------------- #
# math primitives
# --------------------------------------------------------------------------- #
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor,
               alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    # gpt-oss clamped GLU (model.cpp: ggml_swiglu_oai(gate, up, 1.702, 7.0)).
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class PFModel:
    def __init__(self, w: dict[str, torch.Tensor], hp: HP) -> None:
        self.w, self.hp = w, hp
        self.ff, self.attn_factor = yarn_freq_factors(hp)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hp, w = self.hp, self.w
        n = int(ids.shape[0])
        dev = ids.device
        H, Hkv, dh = hp.n_head, hp.n_head_kv, hp.head_dim
        group = H // Hkv
        scale = 1.0 / math.sqrt(dh)

        cos, sin = self._rope_tables(n, dev)         # [n, dh/2]
        mask = self._swa_mask(n, dev)                # [n, n]  0 / -inf
        x = w["tok_embd"][ids]                       # [n, D]

        for i in range(hp.n_layer):
            o = f"l{i}."
            resid = x
            h = rmsnorm(x, w[o + "attn_norm"], hp.rms_eps)

            q = (h @ w[o + "wq"].T + w[o + "bq"]).view(n, H, dh)
            k = (h @ w[o + "wk"].T + w[o + "bk"]).view(n, Hkv, dh)
            v = (h @ w[o + "wv"].T + w[o + "bv"]).view(n, Hkv, dh)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            k = k.repeat_interleave(group, dim=1)    # GQA: kv head shared by `group` q heads
            v = v.repeat_interleave(group, dim=1)
            q = q.permute(1, 0, 2)                    # [H, n, dh]
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

            scores = torch.matmul(q, k.transpose(1, 2)) * scale + mask        # [H, n, n]
            sink = w[o + "sinks"].view(H, 1, 1)                               # value-less extra column
            m = torch.maximum(scores.amax(-1, keepdim=True), sink)
            e = torch.exp(scores - m)
            attn = e / (e.sum(-1, keepdim=True) + torch.exp(sink - m))
            ao = torch.matmul(attn, v).permute(1, 0, 2).reshape(n, H * dh)    # [n, H*dh]
            x = resid + (ao @ w[o + "wo"].T + w[o + "bo"])

            resid = x
            h = rmsnorm(x, w[o + "post_norm"], hp.rms_eps)
            x = resid + self._moe(h, o)

        x = rmsnorm(x, w["output_norm"], hp.rms_eps)
        return x @ w["cls_w"].T + w["cls_b"]                                  # [n, n_cls]

    def _moe(self, x: torch.Tensor, o: str) -> torch.Tensor:
        # Gather form — matches model.cpp exactly: top-k of 128, softmax over the
        # k selected logits (the HF /k and *k cancel), weighted expert sum.
        hp, w = self.hp, self.w
        rl = x @ w[o + "router_w"].T + w[o + "router_b"]          # [n, n_expert]
        topv, topi = torch.topk(rl, hp.n_expert_used, dim=-1)     # [n, k]
        gw = torch.softmax(topv, dim=-1)
        out = torch.zeros_like(x)
        for s in range(hp.n_expert_used):
            e = topi[:, s]                                        # [n] expert id per token
            gate = torch.einsum("ni,noi->no", x, w[o + "gate_w"][e]) + w[o + "gate_b"][e]
            up = torch.einsum("ni,noi->no", x, w[o + "up_w"][e]) + w[o + "up_b"][e]
            hh = swiglu_oai(gate, up)
            oe = torch.einsum("ni,noi->no", hh, w[o + "down_w"][e]) + w[o + "down_b"][e]
            out = out + gw[:, s:s + 1] * oe
        return out

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # interleaved (GPT-J) pairs (2i, 2i+1) — model.cpp mode 0, NOT NeoX.
        x0, x1 = x[..., 0::2], x[..., 1::2]
        c, s = cos[:, None, :], sin[:, None, :]
        out = torch.empty_like(x)
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
        return out

    def _rope_tables(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hp = self.hp
        pos = torch.arange(n, dtype=torch.float32, device=device)
        j = torch.arange(hp.head_dim // 2, dtype=torch.float32, device=device)
        inv = (hp.rope_base ** (-2.0 * j / hp.head_dim)) / self.ff.to(device)
        theta = pos[:, None] * inv[None, :]
        return torch.cos(theta) * self.attn_factor, torch.sin(theta) * self.attn_factor

    def _swa_mask(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n, device=device)
        d = (idx[None, :] - idx[:, None]).abs()
        return torch.where(d <= self.hp.swa_radius, 0.0, float("-inf"))


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def main() -> int:
    if len(sys.argv) < 3:
        sys.exit("usage: pf_model.py <model_dir> <text>")
    model_dir, text = Path(sys.argv[1]), sys.argv[2]

    hp = load_hp(model_dir)
    model = PFModel(load_weights(model_dir, hp), hp)

    from tokenizers import Tokenizer
    enc = Tokenizer.from_file(str(model_dir / "tokenizer.json")).encode(text)
    ids = torch.tensor(enc.ids, dtype=torch.long)

    logits = model.forward(ids)
    pred = logits.argmax(-1).tolist()

    print(f"{len(enc.ids)} tokens · {hp.n_cls} labels · swa_radius={hp.swa_radius}\n")
    for pid, (a, b) in zip(pred, enc.offsets):
        lab = hp.labels[pid]
        if lab != "O":
            print(f"  {text[a:b]!r:24} -> {lab}")
    if all(hp.labels[p] == "O" for p in pred):
        print("  (no PII predicted — check tokenizer specials / weight mapping)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
