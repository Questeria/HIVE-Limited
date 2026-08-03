"""HIVE Limited — a minimal batch-1 decode engine.

Purpose-written for this repository. It is not a stripped copy of the HIVE driver: a
stripped copy leaves the shape of what was removed visible in its own structure, and one
missed branch leaks a lane. This file can only do what it says.

What it does:  load weights -> quantize to the certified per-32 int4 form -> resident
batch-1 decode -> greedy sampling.

What it does not do, anywhere, by any flag: batching, compressed KV, multi-GPU,
serving. Those are absent rather than disabled (see CONTENTS.md).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

F32 = np.float32

# ---------------------------------------------------------------------------
# Quantization — the certified per-32 int4 weight form
# ---------------------------------------------------------------------------
# This is the ONE numeric contract this engine must reproduce exactly, because the
# reproducibility check in verify.sh compares against an independent reference computed
# from these same quantized values. It was byte-verified during packaging.
#
#   per 32-element block along K:  s = max|w| / 7   (s == 0 -> 1.0)
#                                  q = clip(round(w / s), -7, +7)      int8
#   packing (byte-sliced, 8 values per 32-bit word):
#                                  word i, byte b  =  q[8i+b] in the low nibble
#                                                  |  q[8i+4+b] in the high nibble
#   scales: [N, K/32] float32

BLOCK = 32


def quantize_w4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense [N, K] float weights -> (packed uint32 [N, K/8], scales float32 [N, K/32])."""
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {w.shape}")
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"K={k} is not a multiple of {BLOCK}")

    blocks = w.astype(F32).reshape(n, k // BLOCK, BLOCK)
    scale = (np.abs(blocks).max(2) / F32(7.0)).astype(F32)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(blocks / scale[:, :, None]), -7, 7).astype(np.int8).reshape(n, k)

    # low nibble of each byte holds q[8i+b]; high nibble holds q[8i+4+b]
    g = (q.view(np.uint8) & 15).astype(np.uint32).reshape(n, k // 8, 8)
    packed = (g[:, :, 0] | (g[:, :, 1] << 8) | (g[:, :, 2] << 16) | (g[:, :, 3] << 24)
              | (g[:, :, 4] << 4) | (g[:, :, 5] << 12) | (g[:, :, 6] << 20)
              | (g[:, :, 7] << 28)).copy()
    return packed, np.ascontiguousarray(scale)


def dequantize_w4(packed: np.ndarray, scale: np.ndarray, k: int) -> np.ndarray:
    """Inverse of quantize_w4 — used by the reference path in verify.sh.

    Written from the packing rule rather than by inverting the code above, so that a
    mistake in one does not hide in the other.
    """
    n = packed.shape[0]
    out = np.zeros((n, k), np.int8)
    for b in range(4):
        lo = ((packed >> (8 * b)) & 0xF).astype(np.uint8)
        hi = ((packed >> (8 * b + 4)) & 0xF).astype(np.uint8)
        out[:, b::8] = np.where(lo >= 8, lo.astype(np.int16) - 16, lo).astype(np.int8)
        out[:, 4 + b::8] = np.where(hi >= 8, hi.astype(np.int16) - 16, hi).astype(np.int8)
    return out.astype(F32).reshape(n, k // BLOCK, BLOCK).__mul__(
        scale[:, :, None]).reshape(n, k).astype(F32)


# ---------------------------------------------------------------------------
# safetensors — a minimal reader, so this engine needs neither torch nor transformers
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I8": np.int8, "U8": np.uint8, "I32": np.int32, "I64": np.int64,
}


class SafeTensors:
    """Read-only view over a .safetensors file. Header is JSON; tensors are raw bytes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            self.header = json.loads(fh.read(header_len).decode("utf-8"))
        self._data_start = 8 + header_len

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype_name = meta["dtype"]
        if dtype_name not in _ST_DTYPES:
            raise ValueError(f"{name}: unsupported dtype {dtype_name}")
        begin, end = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self._data_start + begin)
            raw = fh.read(end - begin)
        arr = np.frombuffer(raw, dtype=_ST_DTYPES[dtype_name]).reshape(meta["shape"])
        if dtype_name == "BF16":
            # bf16 is the top 16 bits of an f32 — widen rather than lose the exponent
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def load_model_tensors(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Load every tensor from a directory of .safetensors shards."""
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no .safetensors files in {model_dir}\n"
            "HIVE Limited reads a standard safetensors model directory."
        )
    tensors: dict[str, np.ndarray] = {}
    for shard in shards:
        st = SafeTensors(shard)
        for key in st.keys():
            tensors[key] = st.get(key)
    return tensors


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class Config:
    """Geometry read from the model's own config.json — nothing hardcoded per model."""

    def __init__(self, cfg: dict):
        self.n_layer = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        self.ffn = cfg["intermediate_size"]
        self.n_head = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden // self.n_head)
        self.vocab = cfg["vocab_size"]
        self.rms_eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 1000000.0)
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.eos = cfg.get("eos_token_id", None)
        # HIVE Limited caps context deliberately (CONTENTS.md). This is an omission,
        # not a throttle: the KV cache below is simply not allocated any larger.
        self.max_context = 4096

    @property
    def group(self) -> int:
        return self.n_head // self.n_kv

    @classmethod
    def load(cls, model_dir) -> "Config":
        with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as fh:
            return cls(json.load(fh))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

import cudrv  # noqa: E402  (kept last: it dlopen's libcuda at import)

_KERNELS = {
    "k_devgen.ptx": ["gather_emb", "rope_pos", "kvapp_pos", "fda_warp_pos",
                     "argmax_p", "argmax_f"],
    "k_rmsbp.ptx": ["rmsnorm_bp"],
    "k_qa8s.ptx": ["quant_a8s"],
    "k_a4gv.ptx": ["gemv_wpo_a4"],
    "k_silu.ptx": ["gpu_silu_mul"],
    "k_vadd.ptx": ["vec_add"],
}

GVBD = 256          # GEMV block width: one warp per output row, 8 rows per block


class Engine:
    """Batch-1 resident decode. One code path — prefill runs the same step per token.

    That is a real simplification, not a hidden cost centre: prefill is explicitly not a
    benchmarked axis here (README), so it gets the simple correct implementation rather
    than a second kernel family.
    """

    def __init__(self, model_dir, kernels_dir="kernels", verify_kernels=True, quiet=False):
        self.cfg = Config.load(model_dir)
        self.quiet = quiet
        cudrv.init()
        self.dev = cudrv.device(0)
        major, minor = cudrv.compute_capability(self.dev)
        if (major, minor) < (8, 6):
            raise RuntimeError(
                f"HIVE Limited needs compute capability 8.6 or newer; this device is "
                f"{major}.{minor}"
            )
        self.device_name = cudrv.device_name(self.dev)
        self.ctx = cudrv.retain_primary_context(self.dev)
        self._buffers: list[int] = []
        self.fn = self._load_kernels(Path(kernels_dir), verify_kernels)
        self._upload_weights(model_dir)
        self._alloc_state()

    # -- setup ------------------------------------------------------------

    def _log(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    def _load_kernels(self, kdir: Path, verify: bool) -> dict:
        import hashlib
        manifest = {}
        mpath = kdir / "MANIFEST.json"
        if mpath.exists():
            manifest = {e["file"]: e["sha256"]
                        for e in json.loads(mpath.read_text())["kernels"]}
        fns = {}
        for fname, entries in _KERNELS.items():
            blob = (kdir / fname).read_bytes()
            if verify and fname in manifest:
                got = hashlib.sha256(blob).hexdigest()
                if got != manifest[fname]:
                    raise RuntimeError(
                        f"{fname}: sha256 mismatch.\n  manifest {manifest[fname]}\n"
                        f"  on disk  {got}\nRefusing to run kernels that are not the "
                        "ones shipped."
                    )
            mod = cudrv.load_module(blob)
            for e in entries:
                fns[e] = cudrv.get_function(mod, e)
        return fns

    def _alloc(self, nbytes: int) -> int:
        p = cudrv.alloc(nbytes)
        self._buffers.append(p)
        return p

    def _up(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        p = self._alloc(arr.nbytes)
        cudrv.upload(p, arr.ctypes.data, arr.nbytes)
        return p

    def _up_w4(self, w: np.ndarray) -> tuple[int, int, int, int]:
        """Quantize to the certified per-32 int4 form and upload. -> (b4, sw, K, N)"""
        packed, scale = quantize_w4(w)
        return self._up(packed), self._up(scale), w.shape[1], w.shape[0]

    def _upload_weights(self, model_dir):
        c = self.cfg
        t = load_model_tensors(model_dir)
        qd, nq, nkv = c.head_dim, c.n_head, c.n_kv

        def take(name):
            if name not in t:
                raise KeyError(f"missing tensor {name!r} — is this a Qwen3 model?")
            return t[name].astype(F32)

        self._log(f"quantizing {c.n_layer} layers to per-32 int4 ...")
        emb = take("model.embed_tokens.weight")
        self.d_emb16 = self._up(emb.astype(np.float16))          # gather_emb reads f16

        self.layers = []
        for l in range(c.n_layer):
            p = f"model.layers.{l}."
            # qkv is fused into one GEMV: rows [q | k | v]
            wqkv = np.concatenate([take(p + "self_attn.q_proj.weight"),
                                   take(p + "self_attn.k_proj.weight"),
                                   take(p + "self_attn.v_proj.weight")], 0)
            wgu = np.concatenate([take(p + "mlp.gate_proj.weight"),
                                  take(p + "mlp.up_proj.weight")], 0)
            self.layers.append({
                "an": self._up(take(p + "input_layernorm.weight")),
                "fn": self._up(take(p + "post_attention_layernorm.weight")),
                "qn": self._up(take(p + "self_attn.q_norm.weight")),
                "kn": self._up(take(p + "self_attn.k_norm.weight")),
                "qkv": self._up_w4(wqkv),
                "o": self._up_w4(take(p + "self_attn.o_proj.weight")),
                "gu": self._up_w4(wgu),
                "down": self._up_w4(take(p + "mlp.down_proj.weight")),
            })
            if (l + 1) % 7 == 0:
                self._log(f"  {l + 1}/{c.n_layer}")

        self.d_onorm = self._up(take("model.norm.weight"))
        head = emb if c.tied else take("lm_head.weight")
        self.head = self._up_w4(head)
        self.n_head_rows = head.shape[0]  # vocab rows of the output projection
        del t

    def _alloc_state(self):
        c = self.cfg
        hid, ffn, qd = c.hidden, c.ffn, c.head_dim
        nq, nkv, cap = c.n_head, c.n_kv, c.max_context
        qkv_rows = (nq + 2 * nkv) * qd
        wide = max(hid, ffn)

        self.d_h = self._alloc(hid * 4)
        self.d_hn = self._alloc(hid * 4)
        self.d_qkv = self._alloc(qkv_rows * 4)
        self.d_ctx = self._alloc(nq * qd * 4)
        self.d_o = self._alloc(hid * 4)
        self.d_gu = self._alloc(2 * ffn * 4)
        self.d_act = self._alloc(ffn * 4)
        self.d_logits = self._alloc(self.n_head_rows * 4)
        # graph-replay state; _stream_cur is the stream every launch targets (0 = default)
        self._stream_cur = 0
        self._stream = 0
        self._graph = 0
        self._graph_exec = 0
        self._graph_ok = None        # None = untested, True/False = verified
        self.d_a8 = self._alloc(wide)                    # int8 activations
        self.d_sa = self._alloc(2 * (wide // BLOCK) * 4)  # scale + block-sum halves

        # KV cache: f32 only. HIVE Limited ships no compressed-KV lane (CONTENTS.md).
        # This is the sharpest deliberate limitation: fine for short batch-1 chat,
        # hopeless for long context or concurrency — which is exactly the serving axis.
        self.kv_bytes = 2 * c.n_layer * nkv * cap * qd * 4
        self.KC = [self._alloc(nkv * cap * qd * 4) for _ in range(c.n_layer)]
        self.VC = [self._alloc(nkv * cap * qd * 4) for _ in range(c.n_layer)]

        half = qd // 2
        pos = np.arange(cap, dtype=np.float64)[:, None]
        inv = 1.0 / (c.rope_theta ** (np.arange(half, dtype=np.float64) * 2.0 / qd))
        ang = pos * inv[None, :]
        self.d_cos = self._up(np.cos(ang).astype(F32))
        self.d_sin = self._up(np.sin(ang).astype(F32))
        self.d_scale = self._up(np.array([1.0 / np.sqrt(qd)], F32))

        self.d_tok = self._alloc(4)
        self.d_pos = self._alloc(4)
        self.d_step = self._alloc(4)
        self.d_pv = self._alloc(128 * 4)
        self.d_pi = self._alloc(128 * 4)
        self.d_gen = self._alloc(c.max_context * 4)

    def _set_i32(self, ptr: int, value: int):
        cudrv.upload(ptr, np.array([value], np.int32).ctypes.data, 4)

    def _get_i32(self, ptr: int) -> int:
        out = np.zeros(1, np.int32)
        cudrv.download(out.ctypes.data, ptr, 4)
        return int(out[0])

    # -- the decode step --------------------------------------------------

    def _launch(self, name, grid, block, args):
        # Every kernel in the step goes to whichever stream is current. Capture flips this
        # once rather than threading a stream argument through ~30 call sites, which is
        # also why capture cannot half-apply: there is one switch.
        cudrv.launch(self.fn[name], grid, block, args, stream=self._stream_cur)

    def _rmsnorm(self, src, weight, dst, rows, cols):
        self._launch("rmsnorm_bp", rows, 256, [src, dst, weight, cudrv.Scalar(cols)])

    def _gemv(self, src, w4, dst):
        """quantize activations to per-32 int8, then the exact-int32 int4 GEMV."""
        b4, sw, k, n = w4
        nb = k // BLOCK
        self._launch("quant_a8s", (nb + 7) // 8, 256,
                     [src, self.d_a8, self.d_sa, cudrv.Scalar(nb)])
        self._launch("gemv_wpo_a4", (n * 32 + GVBD - 1) // GVBD, GVBD,
                     [self.d_a8, self.d_sa, b4, sw, dst,
                      cudrv.Scalar(k // 8), cudrv.Scalar(n)])

    def _step(self):
        """One full transformer step at the position currently in d_pos."""
        c = self.cfg
        hid, ffn, qd = c.hidden, c.ffn, c.head_dim
        nq, nkv, cap = c.n_head, c.n_kv, c.max_context
        dq = self.d_qkv
        dk = self.d_qkv + nq * qd * 4
        dv = self.d_qkv + (nq + nkv) * qd * 4
        half, grp, kvstride = qd // 2, c.group, cap * qd

        self._launch("gather_emb", (hid + 255) // 256, 256,
                     [self.d_emb16, self.d_tok, self.d_h, cudrv.Scalar(hid)])

        for l, w in enumerate(self.layers):
            self._rmsnorm(self.d_h, w["an"], self.d_hn, 1, hid)
            self._gemv(self.d_hn, w["qkv"], self.d_qkv)

            # Qwen3 normalises q and k per head before RoPE
            self._rmsnorm(dq, w["qn"], dq, nq, qd)
            self._rmsnorm(dk, w["kn"], dk, nkv, qd)
            self._launch("rope_pos", nq, 256,
                         [dq, self.d_cos, self.d_sin, self.d_pos, cudrv.Scalar(half)])
            self._launch("rope_pos", nkv, 256,
                         [dk, self.d_cos, self.d_sin, self.d_pos, cudrv.Scalar(half)])
            self._launch("kvapp_pos", nkv, qd,
                         [dk, dv, self.KC[l], self.VC[l], self.d_pos,
                          cudrv.Scalar(qd), cudrv.Scalar(cap)])
            self._launch("fda_warp_pos", nq, 32,
                         [dq, self.KC[l], self.VC[l], self.d_ctx, self.d_scale,
                          self.d_pos, cudrv.Scalar(qd), cudrv.Scalar(0),
                          cudrv.Scalar(grp), cudrv.Scalar(kvstride)])

            self._gemv(self.d_ctx, w["o"], self.d_o)
            self._launch("vec_add", (hid + 255) // 256, 256,
                         [self.d_h, self.d_o, self.d_h, cudrv.Scalar(hid)])

            self._rmsnorm(self.d_h, w["fn"], self.d_hn, 1, hid)
            self._gemv(self.d_hn, w["gu"], self.d_gu)
            self._launch("gpu_silu_mul", (ffn + 255) // 256, 256,
                         [self.d_gu, self.d_gu + ffn * 4, self.d_act, cudrv.Scalar(ffn)])
            self._gemv(self.d_act, w["down"], self.d_o)
            self._launch("vec_add", (hid + 255) // 256, 256,
                         [self.d_h, self.d_o, self.d_h, cudrv.Scalar(hid)])

        self._rmsnorm(self.d_h, self.d_onorm, self.d_hn, 1, hid)
        self._gemv(self.d_hn, self.head, self.d_logits)

    def _argmax_into_next(self):
        """Pick the next token and advance tok/pos/step entirely on device."""
        self._launch("argmax_p", 128, 256,
                     [self.d_logits, self.d_pv, self.d_pi,
                      cudrv.Scalar(self.n_head_rows), cudrv.Scalar(128)])
        self._launch("argmax_f", 1, 128,
                     [self.d_pv, self.d_pi, self.d_tok, self.d_gen,
                      self.d_step, self.d_pos, cudrv.Scalar(128)])

    def bind_thread(self):
        """Bind this engine's CUDA context to the calling thread.

        Required before any GPU call from a thread that did not construct the engine —
        e.g. an HTTP handler thread. Found by end-to-end test, not by reading docs.
        """
        cudrv.set_current(self.ctx)

    # -- public API -------------------------------------------------------

    def generate(self, prompt_ids, max_new=64, stop_on_eos=True, use_graph=None):
        """Greedy generation. Returns (tokens, prefill_seconds, decode_seconds)."""
        import time
        c = self.cfg
        if not prompt_ids:
            raise ValueError("prompt_ids is empty")
        if len(prompt_ids) + max_new > c.max_context:
            raise ValueError(
                f"{len(prompt_ids)} prompt + {max_new} new exceeds the {c.max_context}-token "
                "context. HIVE Limited caps context deliberately (see CONTENTS.md)."
            )

        self._set_i32(self.d_step, 0)
        cudrv.memset8(self.d_gen, 0, c.max_context * 4)

        # Prefill: the same step, once per prompt token. Unoptimised on purpose —
        # prefill is not a benchmarked axis for this build.
        cudrv.synchronize()
        t0 = time.perf_counter()
        for i, tok in enumerate(prompt_ids):
            self._set_i32(self.d_tok, int(tok))
            self._set_i32(self.d_pos, i)
            self._step()
        cudrv.synchronize()
        t1 = time.perf_counter()

        # Decode: argmax_f advances tok/pos/step on device, so the loop needs no host
        # round-trip per token — which is what makes the throughput number meaningful.
        # Replay the captured step when it is available and proven. `use_graph=None` means
        # "use it if it has passed verification"; the verifier itself passes True/False to
        # compare the two paths.
        graph = self._graph_ok if use_graph is None else use_graph
        if graph:
            try:
                self._build_graph()
            except Exception:
                graph = False          # fall back to issuing directly; never fail the run

        out = []
        for _ in range(max_new):
            if graph:
                self._step_graph()
            else:
                self._argmax_into_next()
                self._step()
            if stop_on_eos:
                cudrv.synchronize()
                tok = self._get_i32(self.d_tok)
                out.append(tok)
                if c.eos is not None and tok == c.eos:
                    break
            else:
                out.append(None)
        cudrv.synchronize()
        t2 = time.perf_counter()

        if out and out[0] is None:                       # untimed-readback path
            got = np.zeros(c.max_context, np.int32)
            cudrv.download(got.ctypes.data, self.d_gen, c.max_context * 4)
            out = [int(x) for x in got[:len(out)]]
        return out, t1 - t0, t2 - t1

    # ---- graph-replayed decode ------------------------------------------------
    #
    # MEASURED, not assumed: one decode step issues 538 kernel launches, and each costs
    # ~25 us of HOST time through ctypes — 13.7 ms of a 16.8 ms step spent in Python while
    # the GPU waits. The kernels were never the problem.
    #
    # A CUDA graph records that sequence once and replays it with a single driver call.
    # This works here only because the step is already host-free by design: argmax writes
    # the next token, position and step counter ON THE DEVICE, so a replay advances the
    # state exactly as a fresh issue would. That property was in the engine before graphs
    # were; graphs just cash it in.
    #
    # The graph is verified against the eager path before it is trusted — see verify_graph.

    def _build_graph(self):
        if self._graph_exec:
            return
        self._stream = cudrv.stream_create()
        cudrv.synchronize()
        cudrv.capture_begin(self._stream)
        self._stream_cur = self._stream
        try:
            self._argmax_into_next()
            self._step()
            graph = cudrv.capture_end(self._stream)
        except Exception:
            try:
                cudrv.capture_end(self._stream)
            except Exception:
                pass
            raise
        finally:
            self._stream_cur = 0
        self._graph = graph
        self._graph_exec = cudrv.graph_instantiate(graph)

    def _step_graph(self):
        cudrv.graph_launch(self._graph_exec, self._stream)

    def verify_graph(self, prompt_ids, n=24) -> bool:
        """Prove the graph produces the SAME tokens as issuing the launches directly.

        A graph that silently diverged would make the engine fast and wrong, which is worse
        than slow — so this runs before the fast path is used, and a mismatch disables it
        rather than being reported as a curiosity.
        """
        eager, _, _ = self.generate(prompt_ids, max_new=n, use_graph=False)
        fast, _, _ = self.generate(prompt_ids, max_new=n, use_graph=True)
        return eager == fast

    def stream(self, prompt_ids, max_new=64, stop_evt=None):
        """Greedy generation, yielding each token as it is produced.

        Yields ``("prefill", seconds)`` once, then ``("tok", id, seconds_since_first_token)``
        per token. The arena UI records these offsets and replays the run at its real
        cadence, so they have to be the engine's own timings, taken here, per token — not
        interpolated afterwards from a total. `generate()` above stays as the batch API
        that `bench.py` and `verify.py` use; this is the same loop with the readback that
        was already happening turned into a yield.

        The per-token `cuCtxSynchronize` is what makes an honest timestamp possible and it
        costs a little throughput, so `bench.sh` — not this path — is the number to quote.
        """
        import time
        c = self.cfg
        if not prompt_ids:
            raise ValueError("prompt_ids is empty")
        if len(prompt_ids) + max_new > c.max_context:
            raise ValueError(
                f"{len(prompt_ids)} prompt + {max_new} new exceeds the {c.max_context}-token "
                "context. HIVE Limited caps context deliberately (see CONTENTS.md)."
            )

        self._set_i32(self.d_step, 0)
        cudrv.memset8(self.d_gen, 0, c.max_context * 4)

        cudrv.synchronize()
        t0 = time.perf_counter()
        for i, tok in enumerate(prompt_ids):
            self._set_i32(self.d_tok, int(tok))
            self._set_i32(self.d_pos, i)
            self._step()
        cudrv.synchronize()
        yield ("prefill", time.perf_counter() - t0)

        graph = bool(self._graph_ok)
        if graph:
            try:
                self._build_graph()
            except Exception:
                graph = False

        first = None
        for _ in range(max_new):
            if stop_evt is not None and stop_evt.is_set():
                return
            if graph:
                self._step_graph()
            else:
                self._argmax_into_next()
                self._step()
            cudrv.synchronize()
            now = time.perf_counter()
            if first is None:
                first = now
            tok = self._get_i32(self.d_tok)
            yield ("tok", tok, now - first)
            if c.eos is not None and tok == c.eos:
                return

    def logits(self, prompt_ids) -> np.ndarray:
        """Run the prompt and return the final f32 logits — used by verify.sh."""
        for i, tok in enumerate(prompt_ids):
            self._set_i32(self.d_tok, int(tok))
            self._set_i32(self.d_pos, i)
            self._step()
        cudrv.synchronize()
        out = np.zeros(self.n_head_rows, F32)
        cudrv.download(out.ctypes.data, self.d_logits, out.nbytes)
        return out

    def _weight_bits(self) -> tuple[int, int]:
        """(total bits, total elements) over every quantized matrix actually loaded."""
        bits = elems = 0
        for w in self.layers:
            for key in ("qkv", "o", "gu", "down"):
                b4, sw, k, n = w[key]
                bits += n * (k // 8) * 32 + n * (k // BLOCK) * 32
                elems += n * k
        b4, sw, k, n = self.head
        bits += n * (k // 8) * 32 + n * (k // BLOCK) * 32
        elems += n * k
        return bits, elems

    def bits_per_weight(self) -> float:
        """Measured, not quoted: packed int4 body + f32 per-32 scales, over element count."""
        bits, elems = self._weight_bits()
        return bits / elems

    def weight_bytes(self) -> int:
        """Total weight bytes — what a batch-1 decode step must read per token.

        The arena's fuel gauge uses this instead of a hardcoded rate, so what it draws
        follows the weights that were loaded rather than a figure someone typed once.
        """
        bits, _ = self._weight_bits()
        return bits // 8

    def free(self):
        if getattr(self, "_graph_exec", 0) or getattr(self, "_graph", 0):
            cudrv.graph_destroy(self._graph, self._graph_exec)
            self._graph = self._graph_exec = 0
        for p in self._buffers:
            try:
                cudrv.free(p)
            except Exception:
                pass
        self._buffers.clear()
