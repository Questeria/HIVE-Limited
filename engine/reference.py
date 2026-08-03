"""An independent numpy reference for HIVE Limited.

This exists so the engine's output can be *checked*, not merely asserted. It is written
from the numeric contract rather than from the engine's code, and it runs on the CPU in
numpy — it shares no kernel, no PTX, and no GPU code path with the thing it verifies.

WHAT IT PROVES, precisely:

  * the integer phase is exact. Activations are quantized per 32 with the same rule the
    kernel uses, weights are the same int4 values, and the dot products are int32 — so
    those are equal by construction on both sides, not approximately equal.
  * the tokens agree. The reference and the engine must select the same token at every
    position.

WHAT IT DOES NOT PROVE:

  * that the f32 logits are bit-identical. The engine reduces across 32 lanes in a warp
    butterfly; numpy reduces in its own order. Float addition is not associative, so the
    last bits can differ. Token agreement is the claim; f32 bit-identity is not.
  * conformance to any other implementation of this model. See docs/VERIFY.md — the
    RMSNorm epsilon here matches the shipped kernel (1e-5), which is what makes this a
    check of THIS engine.

Both limits are stated because a verification tool that overstates what it verifies is
worse than none.
"""
from __future__ import annotations

import numpy as np

from limited import BLOCK, dequantize_w4

F32 = np.float32
RMS_EPS = np.float32(1e-5)     # matches rmsnorm_bp in the shipped kernels — see VERIFY.md


def rmsnorm(x: np.ndarray, w: np.ndarray, eps=RMS_EPS) -> np.ndarray:
    """y = w * x * rsqrt(mean(x^2) + eps), row-wise."""
    x = x.astype(F32)
    ms = (x.astype(F32) ** 2).mean(-1, keepdims=True).astype(F32)
    return (w.astype(F32) * (x / np.sqrt(ms + eps))).astype(F32)


def quantize_act(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-32 activation quant, replicating quant_a8s exactly.

    scale s = max|a|/127 (0 -> 1); q = trunc(a/s +- 0.5) clipped to +-127; and the
    per-block signed sum the GEMV's bias correction consumes.
    """
    a = a.astype(F32).reshape(-1)
    nb = a.size // BLOCK
    blocks = a.reshape(nb, BLOCK)
    mx = np.abs(blocks).max(1)
    s = (mx / F32(127.0)).astype(F32)
    s[s == 0] = F32(1.0)
    half = np.where(blocks < 0, F32(-0.5), F32(0.5)).astype(F32)
    q = np.trunc(blocks / s[:, None] + half).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int32)
    return q, s, q.sum(1).astype(np.int32)


def gemv_w4(a: np.ndarray, packed: np.ndarray, scale: np.ndarray, k: int) -> np.ndarray:
    """The engine's GEMV: exact int32 block dots, then an f32 per-block scale combine."""
    q_a, s_a, _ = quantize_act(a)                     # [nb, 32] int32
    qw = _weight_ints(packed, k)                      # [N, K] int32, exact int4 values
    n = qw.shape[0]
    nb = k // BLOCK
    qw_b = qw.reshape(n, nb, BLOCK)
    # integer dots per (row, block) — exact, no rounding anywhere in this line
    idot = np.einsum("nbk,bk->nb", qw_b, q_a, optimize=True).astype(np.int64)
    combined = (scale.astype(F32) * s_a[None, :].astype(F32) * idot.astype(F32))
    return combined.sum(1).astype(F32)


def _weight_ints(packed: np.ndarray, k: int) -> np.ndarray:
    """Unpack to the signed int4 values, without applying scales."""
    n = packed.shape[0]
    out = np.zeros((n, k), np.int32)
    for b in range(4):
        lo = ((packed >> (8 * b)) & 0xF).astype(np.int32)
        hi = ((packed >> (8 * b + 4)) & 0xF).astype(np.int32)
        out[:, b::8] = np.where(lo >= 8, lo - 16, lo)
        out[:, 4 + b::8] = np.where(hi >= 8, hi - 16, hi)
    return out


def rope(vec: np.ndarray, pos: int, cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    """Rotate-half RoPE over [heads, head_dim]."""
    heads, dim = vec.shape
    half = dim // 2
    c, s = cos_t[pos].astype(F32), sin_t[pos].astype(F32)
    lo, hi = vec[:, :half].astype(F32), vec[:, half:].astype(F32)
    return np.concatenate([lo * c - hi * s, hi * c + lo * s], 1).astype(F32)


def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    g = gate.astype(F32)
    sgn = np.where(g < 0, F32(-1.0), F32(1.0)).astype(F32)
    e = np.exp(-(sgn * g)).astype(F32)
    sig = (F32(0.5) * (F32(1.0) + sgn * ((F32(1.0) - e) / (F32(1.0) + e)))).astype(F32)
    return (g * sig * up.astype(F32)).astype(F32)


def attention(q: np.ndarray, kc: np.ndarray, vc: np.ndarray, pos: int,
              group: int, scale: float) -> np.ndarray:
    """Causal attention over the cache up to and including `pos`."""
    n_head, dim = q.shape
    out = np.zeros_like(q, dtype=F32)
    for h in range(n_head):
        kvh = h // group
        keys = kc[kvh, : pos + 1].astype(F32)
        vals = vc[kvh, : pos + 1].astype(F32)
        logits = (keys @ q[h].astype(F32)) * F32(scale)
        logits -= logits.max()
        w = np.exp(logits).astype(F32)
        w /= w.sum()
        out[h] = (w[:, None] * vals).sum(0)
    return out


class ReferenceModel:
    """A complete CPU forward pass, built from the primitives above.

    Slow on purpose — it is a checker, not an engine. It quantizes the weights itself
    with the same published rule, so it depends on none of the GPU state.
    """

    def __init__(self, model_dir, max_context=4096):
        from limited import Config, load_model_tensors, quantize_w4
        self.cfg = Config.load(model_dir)
        c = self.cfg
        self.max_context = max_context
        t = load_model_tensors(model_dir)

        def take(name):
            return t[name].astype(F32)

        self.emb = take("model.embed_tokens.weight").astype(np.float16)   # engine reads f16
        self.layers = []
        for l in range(c.n_layer):
            p = f"model.layers.{l}."
            wqkv = np.concatenate([take(p + "self_attn.q_proj.weight"),
                                   take(p + "self_attn.k_proj.weight"),
                                   take(p + "self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([take(p + "mlp.gate_proj.weight"),
                                  take(p + "mlp.up_proj.weight")], 0)
            self.layers.append({
                "an": take(p + "input_layernorm.weight"),
                "fn": take(p + "post_attention_layernorm.weight"),
                "qn": take(p + "self_attn.q_norm.weight"),
                "kn": take(p + "self_attn.k_norm.weight"),
                "qkv": quantize_w4(wqkv) + (wqkv.shape[1],),
                "o": quantize_w4(take(p + "self_attn.o_proj.weight")) + (c.n_head * c.head_dim,),
                "gu": quantize_w4(wgu) + (wgu.shape[1],),
                "down": quantize_w4(take(p + "mlp.down_proj.weight")) + (c.ffn,),
            })
        self.onorm = take("model.norm.weight")
        head = self.emb.astype(F32) if c.tied else take("lm_head.weight")
        self.head = quantize_w4(head) + (head.shape[1],)
        del t

        qd, half = c.head_dim, c.head_dim // 2
        pos = np.arange(max_context, dtype=np.float64)[:, None]
        inv = 1.0 / (c.rope_theta ** (np.arange(half, dtype=np.float64) * 2.0 / qd))
        ang = pos * inv[None, :]
        self.cos, self.sin = np.cos(ang).astype(F32), np.sin(ang).astype(F32)
        self.reset()

    def reset(self):
        c = self.cfg
        self.KC = np.zeros((c.n_layer, c.n_kv, self.max_context, c.head_dim), F32)
        self.VC = np.zeros((c.n_layer, c.n_kv, self.max_context, c.head_dim), F32)

    def step(self, token: int, pos: int) -> np.ndarray:
        c = self.cfg
        qd, nq, nkv = c.head_dim, c.n_head, c.n_kv
        h = self.emb[token].astype(F32)

        for l, w in enumerate(self.layers):
            hn = rmsnorm(h, w["an"])
            packed, scale, k = w["qkv"]
            qkv = gemv_w4(hn, packed, scale, k)
            q = qkv[: nq * qd].reshape(nq, qd)
            kk = qkv[nq * qd: (nq + nkv) * qd].reshape(nkv, qd)
            v = qkv[(nq + nkv) * qd:].reshape(nkv, qd)

            q = np.stack([rmsnorm(q[i], w["qn"]) for i in range(nq)])
            kk = np.stack([rmsnorm(kk[i], w["kn"]) for i in range(nkv)])
            q = rope(q, pos, self.cos, self.sin)
            kk = rope(kk, pos, self.cos, self.sin)

            self.KC[l][:, pos, :] = kk
            self.VC[l][:, pos, :] = v
            ctx = attention(q, self.KC[l], self.VC[l], pos, c.group,
                            1.0 / np.sqrt(qd)).reshape(-1)

            packed, scale, k = w["o"]
            h = (h + gemv_w4(ctx, packed, scale, k)).astype(F32)

            hn = rmsnorm(h, w["fn"])
            packed, scale, k = w["gu"]
            gu = gemv_w4(hn, packed, scale, k)
            act = silu_mul(gu[: c.ffn], gu[c.ffn:])
            packed, scale, k = w["down"]
            h = (h + gemv_w4(act, packed, scale, k)).astype(F32)

        hn = rmsnorm(h, self.onorm)
        packed, scale, k = self.head
        return gemv_w4(hn, packed, scale, k)

    def generate(self, prompt_ids, max_new=8):
        """Greedy, same rule as the engine: argmax over the final logits."""
        self.reset()
        logits = None
        for i, tok in enumerate(prompt_ids):
            logits = self.step(int(tok), i)
        out, pos = [], len(prompt_ids)
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if self.cfg.eos is not None and nxt == self.cfg.eos:
                break
            logits = self.step(nxt, pos)
            pos += 1
        return out
