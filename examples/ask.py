"""Use the HIVE Limited engine from your own Python program.

    python3 examples/ask.py ~/.hive-limited/models/Qwen3-1.7B "Why is the sky blue?"

This is the whole integration surface: construct an Engine, encode with the bundled
tokenizer, then either generate() for a blocking call or stream() for token-by-token.
Everything runs locally; nothing leaves your machine.

The build stays deliberately limited (see CONTENTS.md): batch-1 only, greedy sampling,
one quantization form. If your application needs batching, sampling temperatures, or a
serving layer, this is not the engine for it — that is the full engine's job, and this
repository does not contain it.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import limited, tokenizer_min  # noqa: E402

KERNELS = Path(__file__).resolve().parent.parent / "kernels"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_dir, prompt = sys.argv[1], sys.argv[2]
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    tokenizer_min.configure(model_dir)
    tok = tokenizer_min.get()
    engine = limited.Engine(model_dir, kernels_dir=str(KERNELS))

    ids = tok.encode(prompt)
    print(f"[{engine.device_name} | {engine.bits_per_weight():.3f} bits/weight]")

    # Blocking form: returns (tokens, prefill_seconds, decode_seconds).
    out, t_pre, t_dec = engine.generate(ids, max_new=max_new)
    text = tok.decode(out)
    n = len(out)
    print(text)
    print(f"\n[{n} tokens | prefill {t_pre:.2f}s | decode {t_dec:.2f}s"
          f" | {n / t_dec:.1f} tok/s measured just now on your machine]")

    # Streaming form: yields ("prefill", seconds) once, then ("tok", id, seconds)
    # per token, timed by the engine itself.
    print("\n--- the same call, streamed ---")
    t0 = time.perf_counter()
    for ev in engine.stream(ids, max_new=32):
        if ev[0] == "tok":
            print(tok.decode([ev[1]]), end="", flush=True)
    print(f"\n[streamed for {time.perf_counter() - t0:.2f}s]")

    engine.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
