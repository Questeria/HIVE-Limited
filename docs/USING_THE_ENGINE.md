# Using the engine in your own application

The demo is the arena page, but the engine underneath it is an ordinary Python class you
can call from your own code. This page is the whole integration story; the runnable
version is [`examples/ask.py`](../examples/ask.py).

**What you get** is exactly what this build ships and nothing more: local batch-1 greedy
decode of Qwen3 dense models on one NVIDIA GPU (compute capability 8.6+), through the
driver alone — importing the engine pulls in no CUDA toolkit, no PyTorch, no framework.
If your application needs batching, sampling temperatures, long-context serving, or
multi-user throughput, this build does not do those and will not grow them (see
`CONTENTS.md` for what is deliberately absent).

## The whole integration

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("hive-limited") / "engine"))   # wherever you cloned it
import limited, tokenizer_min

tokenizer_min.configure(model_dir)                  # once, before get()
tok = tokenizer_min.get()
engine = limited.Engine(model_dir, kernels_dir="hive-limited/kernels")

ids = tok.encode("Why is the sky blue?")
tokens, prefill_s, decode_s = engine.generate(ids, max_new=128)
print(tok.decode(tokens))
```

`Engine()` loads the safetensors weights, quantizes them to the certified per-32 int4
form, verifies every kernel against `kernels/MANIFEST.json` by sha256, and uploads to
the GPU. First construction takes a few seconds; keep the instance and reuse it.

## The API

| call | what it does |
|---|---|
| `Engine(model_dir, kernels_dir="kernels", verify_kernels=True, quiet=False)` | load + quantize + upload. Raises on cc < 8.6, on a missing model, on a kernel hash mismatch |
| `engine.generate(prompt_ids, max_new=64, stop_on_eos=True, use_graph=None)` | blocking greedy generation → `(tokens, prefill_seconds, decode_seconds)` |
| `engine.stream(prompt_ids, max_new=64, stop_evt=None)` | generator: yields `("prefill", seconds)` once, then `("tok", id, seconds_since_first_token)` per token. Pass a `threading.Event` as `stop_evt` to cancel from another thread |
| `engine.logits(prompt_ids)` | one forward pass → the final-position logits as a numpy array |
| `engine.bits_per_weight()` / `engine.weight_bytes()` | what is actually resident: 5.000 bits/weight (4-bit values + one f32 scale per 32) |
| `engine.verify_graph(prompt_ids, n=24)` | proves the CUDA-graph fast path token-identical against the plain path before it is used |
| `engine.free()` | release GPU memory early (the driver reclaims everything at process exit regardless) |
| `tokenizer_min.configure(dir)` / `.get()` | the bundled standard-library Qwen3 tokenizer: `encode(str) -> ids`, `decode(ids) -> str` |

Threading: the engine is single-stream by design — one `generate()`/`stream()` at a
time. If you call it from a thread other than the one that built it, call
`engine.bind_thread()` in that thread first (the demo server does exactly this).

## Two honest notes

- `stream()` synchronizes per token so its timestamps are real; that costs a little
  throughput. The number to quote is `bench.sh`, which measures the batch path.
- Prompts are encoded raw — no chat template — so instruct-style models will behave like
  completion models. That is the demo's behavior too, and it is the honest default for
  an engine this small: a template is a prompt-engineering choice, not an engine feature.
