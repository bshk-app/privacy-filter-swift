#!/usr/bin/env python3
"""ANE-shaped PyTorch variant of the privacy-filter forward (step 2).

Same math as apple/pf_model.py (the SSOT reference) rewritten in the
ml-ane-transformers idiom so Core ML can place it on the Neural Engine:
  - (B, C, 1, S) tensors throughout;
  - every projection is an nn.Conv2d 1x1  (Linear on (B,C,1,S));
  - RMSNorm over the channel axis;
  - attention sinks = one extra, value-less softmax column;
  - MoE computed DENSE (all experts) then combined by a top-4 softmax weight
    vector, so there is NO dynamic expert gather to knock it off the ANE.
    gate/up are plain 1x1 convs (E*inter out-channels); down is a GROUPED 1x1
    conv (groups = n_expert). Only the tiny top-k that builds the weight vector
    can fall back to CPU.

The sequence length W is FIXED at construction and threaded through as a plain
python int (no x.shape[-1] / arange(symbolic) / zeros(symbolic)) so coremltools
sees fully static shapes — both required for a clean convert and ideal for ANE.

Inputs : emb  [1, n_embd, 1, W]   (token-embedding lookup stays on the host)
         mask [1, 1, W, W]        (additive: SWA band + padding)
Output : logits [1, n_cls, 1, W]

Numerically equal to pf_model.PFModel — see the __main__ selftest.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pf_model import HP, swiglu_oai, yarn_freq_factors


def mk_conv(weight: torch.Tensor, bias: torch.Tensor, groups: int = 1) -> nn.Conv2d:
    """1x1 conv from a [out, in_per_group] weight + [out] bias."""
    out_ch, in_pg = weight.shape
    c = nn.Conv2d(in_pg * groups, out_ch, kernel_size=1, groups=groups, bias=True)
    with torch.no_grad():
        c.weight.copy_(weight.reshape(out_ch, in_pg, 1, 1))
        c.bias.copy_(bias)
    return c


def rmsnorm_c(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # x [1, C, 1, S]; normalise over the channel axis.
    return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + eps) * w.view(1, -1, 1, 1)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               heads: int, dh: int, w: int) -> torch.Tensor:
    # x [1, heads, dh, w]; cos/sin [1, 1, dh/2, w]; interleaved (2i, 2i+1) pairs.
    x0, x1 = x[:, :, 0::2, :], x[:, :, 1::2, :]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.stack([o0, o1], dim=3).reshape(1, heads, dh, w)   # interleave back


class PFLayer(nn.Module):
    def __init__(self, w: dict[str, torch.Tensor], o: str, hp: HP, window: int) -> None:
        super().__init__()
        self.hp, self.W = hp, window
        E, I, D = hp.n_expert, hp.n_inter, hp.n_embd
        self.register_buffer("attn_norm", w[o + "attn_norm"])
        self.register_buffer("post_norm", w[o + "post_norm"])
        self.register_buffer("sinks", w[o + "sinks"])
        self.q = mk_conv(w[o + "wq"], w[o + "bq"])
        self.k = mk_conv(w[o + "wk"], w[o + "bk"])
        self.v = mk_conv(w[o + "wv"], w[o + "bv"])
        self.o = mk_conv(w[o + "wo"], w[o + "bo"])
        self.router = mk_conv(w[o + "router_w"], w[o + "router_b"])
        # dense experts: gate/up plain (every expert sees all D inputs),
        # down grouped so expert e mixes only its own inter-channels.
        self.gate = mk_conv(w[o + "gate_w"].reshape(E * I, D), w[o + "gate_b"].reshape(E * I))
        self.up = mk_conv(w[o + "up_w"].reshape(E * I, D), w[o + "up_b"].reshape(E * I))
        self.down = mk_conv(w[o + "down_w"].reshape(E * D, I), w[o + "down_b"].reshape(E * D), groups=E)

    def _attn(self, x: torch.Tensor, mask: torch.Tensor,
              cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hp, W = self.hp, self.W
        H, Hkv, dh = hp.n_head, hp.n_head_kv, hp.head_dim
        group = H // Hkv
        q = rope_apply(self.q(x).view(1, H, dh, W), cos, sin, H, dh, W)
        k = rope_apply(self.k(x).view(1, Hkv, dh, W), cos, sin, Hkv, dh, W)
        v = self.v(x).view(1, Hkv, dh, W)
        k = k.repeat_interleave(group, dim=1)            # GQA -> H heads
        v = v.repeat_interleave(group, dim=1)
        scale = 1.0 / math.sqrt(dh)
        scores = torch.einsum("bhds,bhdt->bhst", q, k) * scale + mask     # [1,H,W,W]
        sink = self.sinks.view(1, H, 1, 1).expand(1, H, W, 1)             # value-less column
        attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :W]
        out = torch.einsum("bhst,bhdt->bhds", attn, v).reshape(1, H * dh, 1, W)
        return self.o(out)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        hp, W = self.hp, self.W
        E, I, D = hp.n_expert, hp.n_inter, hp.n_embd
        rl = self.router(x).view(E, W).transpose(0, 1)                    # [W, E]
        val, idx = torch.topk(rl, hp.n_expert_used, dim=-1)
        sw = torch.softmax(val, dim=-1)
        wdense = torch.zeros(W, E, dtype=x.dtype, device=x.device).scatter(1, idx, sw.to(x.dtype))
        wdense = wdense.transpose(0, 1).view(1, E, 1, W)                  # [1,E,1,W], 4 nonzeros/col
        h = swiglu_oai(self.gate(x), self.up(x))                         # [1, E*I, 1, W]
        down = self.down(h).view(1, E, D, W)                            # [1, E, D, W]
        return (wdense.view(1, E, 1, W) * down).sum(dim=1).view(1, D, 1, W)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
        x = x + self._attn(rmsnorm_c(x, self.attn_norm, eps), mask, cos, sin)
        x = x + self._moe(rmsnorm_c(x, self.post_norm, eps))
        return x


class PFANE(nn.Module):
    def __init__(self, w: dict[str, torch.Tensor], hp: HP, window: int) -> None:
        super().__init__()
        self.hp, self.W = hp, window
        self.eps = hp.rms_eps
        ff, self.attn_factor = yarn_freq_factors(hp)
        self.register_buffer("ff", ff)
        self.layers = nn.ModuleList([PFLayer(w, f"l{i}.", hp, window) for i in range(hp.n_layer)])
        self.register_buffer("output_norm", w["output_norm"])
        self.cls = mk_conv(w["cls_w"], w["cls_b"])

    def _rope(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        hp, W = self.hp, self.W
        pos = torch.arange(W, dtype=torch.float32, device=device)
        j = torch.arange(hp.head_dim // 2, dtype=torch.float32, device=device)
        inv = (hp.rope_base ** (-2.0 * j / hp.head_dim)) / self.ff.to(device)
        theta = inv[:, None] * pos[None, :]                              # [dh/2, W]
        cos = (torch.cos(theta) * self.attn_factor).to(dtype).view(1, 1, -1, W)
        sin = (torch.sin(theta) * self.attn_factor).to(dtype).view(1, 1, -1, W)
        return cos, sin

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope(emb.device, emb.dtype)
        x = emb
        for layer in self.layers:
            x = layer(x, mask, cos, sin, self.eps)
        return self.cls(rmsnorm_c(x, self.output_norm, self.eps))


# --------------------------------------------------------------------------- #
# host-side helpers (used by export_coreml.py / dump_reference.py)
# --------------------------------------------------------------------------- #
def build_mask(real_len: int, window: int, radius: int, neg: float = -1e4) -> torch.Tensor:
    """Additive [1,1,W,W] mask: visible iff |q-k|<=radius AND both tokens real."""
    idx = torch.arange(window)
    band = (idx[None, :] - idx[:, None]).abs() <= radius
    real = idx < real_len
    vis = band & real[None, :] & real[:, None]
    return torch.where(vis, 0.0, neg).view(1, 1, window, window)


def embed(w: dict[str, torch.Tensor], ids: torch.Tensor, window: int) -> torch.Tensor:
    """Host-side token-embedding lookup -> [1, n_embd, 1, W], zero-padded to window."""
    D = w["tok_embd"].shape[1]
    emb = torch.zeros(window, D)
    emb[: ids.shape[0]] = w["tok_embd"][ids]
    return emb.t().contiguous().view(1, D, 1, window)


# --------------------------------------------------------------------------- #
# selftest: PFANE must equal pf_model.PFModel on random tiny weights
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    from pf_model import PFModel

    torch.manual_seed(0)
    hp = HP(n_layer=2, n_embd=8, n_head=4, n_head_kv=2, head_dim=4, n_inter=8,
            n_expert=6, n_expert_used=2, swa_radius=3, n_ctx_orig=64,
            rms_eps=1e-5, rope_base=150000.0, yarn_factor=32.0,
            yarn_beta_fast=32.0, yarn_beta_slow=1.0, yarn_truncate=False,
            labels=["O", "S-x"])
    H, dh, Hkv, D, E, I, V = hp.n_head, hp.head_dim, hp.n_head_kv, hp.n_embd, hp.n_expert, hp.n_inter, 20
    R = lambda *s: torch.randn(*s) * 0.1
    w = {"tok_embd": R(V, D), "output_norm": R(D), "cls_w": R(hp.n_cls, D), "cls_b": R(hp.n_cls)}
    for i in range(hp.n_layer):
        o = f"l{i}."
        w[o + "attn_norm"] = R(D); w[o + "post_norm"] = R(D); w[o + "sinks"] = R(H)
        w[o + "wq"] = R(H * dh, D); w[o + "bq"] = R(H * dh)
        w[o + "wk"] = R(Hkv * dh, D); w[o + "bk"] = R(Hkv * dh)
        w[o + "wv"] = R(Hkv * dh, D); w[o + "bv"] = R(Hkv * dh)
        w[o + "wo"] = R(D, H * dh); w[o + "bo"] = R(D)
        w[o + "router_w"] = R(E, D); w[o + "router_b"] = R(E)
        w[o + "gate_w"] = R(E, I, D); w[o + "gate_b"] = R(E, I)
        w[o + "up_w"] = R(E, I, D); w[o + "up_b"] = R(E, I)
        w[o + "down_w"] = R(E, D, I); w[o + "down_b"] = R(E, D)

    n = 10
    ref = PFModel(w, hp)
    ane = PFANE(w, hp, n)
    ane.train(False)
    ids = torch.arange(1, 1 + n)
    ref_out = ref.forward(ids)                                    # [n, n_cls]
    emb = w["tok_embd"][ids].t().contiguous().view(1, D, 1, n)
    mask = ref._swa_mask(n, torch.device("cpu")).view(1, 1, n, n)  # exact -inf band (match ref)
    with torch.no_grad():
        ane_out = ane(emb, mask).view(hp.n_cls, n).t()           # [n, n_cls]

    max_abs = (ref_out - ane_out).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ref_out.flatten(), ane_out.flatten(), dim=0).item()
    print(f"max|ref-ane| = {max_abs:.3e}   cosine = {cos:.8f}")
    ok = max_abs < 1e-4 and cos > 0.9999
    print("ANE_SELFTEST_PASS" if ok else "ANE_SELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
