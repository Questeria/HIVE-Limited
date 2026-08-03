"""Measure this build, on this machine, right now.

Every number printed was produced by this run. Nothing is read from a file, and there is
no table of results committed anywhere in this repository — if a figure cannot be
re-derived by the command that prints it, it does not belong in a demo.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
import numpy as np                                     # noqa: E402
import limited, tokenizer_min                          # noqa: E402

PROMPTS = ["The capital of France is",
           "Explain gravity to a five-year-old.",
           "def quicksort(arr):"]


def main(argv):
    if len(argv) < 2:
        print("usage: python bench.py <model_dir> [tokens_per_prompt]")
        return 2
    model_dir, n_new = argv[1], int(argv[2]) if len(argv) > 2 else 64
    tokenizer_min.configure(model_dir)
    tok = tokenizer_min.get()

    t0 = time.perf_counter()
    eng = limited.Engine(model_dir)
    load_s = time.perf_counter() - t0

    print(f"\ndevice        : {eng.device_name}")
    print(f"load+quantize : {load_s:.1f} s")
    print(f"bits/weight   : {eng.bits_per_weight():.3f}   "
          f"(4 body + f32 scale per {limited.BLOCK} weights)")
    print(f"KV cache      : {eng.kv_bytes / 2**30:.2f} GiB reserved at "
          f"{eng.cfg.max_context} ctx, f32 only\n")

    rates = []
    for p in PROMPTS:
        ids = tok.encode(p)
        out, prefill_s, decode_s = eng.generate(ids, max_new=n_new)
        rate = len(out) / decode_s
        rates.append(rate)
        print(f"  {rate:7.1f} tok/s   {len(out):3d} tokens   "
              f"prefill {prefill_s*1e3:6.1f} ms   {p!r}")

    print(f"\nmedian decode : {sorted(rates)[len(rates)//2]:.1f} tok/s")
    print("\nMeasured just now, on this machine. Prefill is an unoptimised path in this")
    print("build and is deliberately not a benchmarked axis (see README).")
    eng.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
