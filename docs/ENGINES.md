# Racing HIVE Limited against other engines

The arena runs HIVE Limited beside another engine on **your** machine and shows both
answering the same prompt. Nothing is pre-recorded: every number on that page was produced
by the run you just watched, on both lanes.

That is deliberate. A benchmark table in a vendor's repository is a claim. A race you ran
yourself is a measurement.

---

## Quick start (the supported defaults)

```bash
./setup/download_models.sh        # Qwen3-1.7B, both formats (~4 GB)
./setup/install_llamacpp.sh       # builds llama.cpp with CUDA (~5 min)
python3 serve.py ~/.hive-limited/models/Qwen3-1.7B
```

Open <http://127.0.0.1:8080>, pick **llama.cpp** in the dropdown, press LAUNCH.

vLLM is also supported and is a heavier install (several GB, brings PyTorch):

```bash
./setup/install_vllm.sh
```

Everything installs under `~/.hive-limited/` and nothing outside it is touched. Override
with `HIVE_LIMITED_HOME`, `HIVE_LIMITED_ENGINES`, `HIVE_LIMITED_MODELS`.

The dropdown labels each rival **— not installed** until it is present, so you can see what
will actually run before you press LAUNCH. Selecting a missing engine tells you how to
install it rather than failing silently.

---

## What the race does and does not measure

**Model loading is never timed.** Cold start shows as a loading bar and is excluded from
both tokens/sec and first-token latency; the clocks start when an engine begins generating.
Timing an engine's startup as if it were inference is the commonest way these comparisons
get rigged, so it is worth stating that we do not.

**One clock, one machine.** Both lanes are timed by the server, so the comparison is not
between two different stopwatches.

**They run one at a time.** Two engines will not fit on a small card, so each runs alone and
the UI replays them together afterwards at their real recorded cadence. The race you watch
is a synchronised replay of two real runs, not a staged animation.

**Both run greedy** (temperature 0), so the race is not a lottery.

**Bit-width is not equal, and that matters.** HIVE Limited runs int4 with a per-32 scale =
**5.00 bits/weight**, computed from the weights actually loaded. llama.cpp's Q4_K_M is
roughly 5.0–5.3; an fp16 vLLM model is 16. Decode is memory-bound, so bits/weight is most of
what decides tokens/sec. **A win against a 16-bit model is mostly a statement about the
format.** To isolate the engine, race a rival at a comparable bit-width — the arena will not
do that reasoning for you, and the note under the race says so.

---

## Bring your own engine

The adapters are small and deliberately so. Any engine exposing an **OpenAI-compatible
streaming API** works — that is most of them, including SGLang, TGI, LM Studio, Ollama
(with its OpenAI endpoint), and llama.cpp or vLLM you installed yourself.

### Point at an engine you are already running

Start it, then start the arena. If a server is already answering on the port a rival uses,
the arena uses it as-is and never tries to launch or stop it:

| rival | port |
|---|---|
| llama.cpp | 8081 |
| vLLM | 8082 |
| TensorRT-LLM (bring your own: `trtllm-serve <model> --port 8083`) | 8083 |

So a llama.cpp you built yourself, with your own flags, is used simply by serving it on
8081 before launching the arena. Your flags, your build, your result.

### Add a new engine

Add an entry to `REGISTRY` in [`engine/rivals.py`](../engine/rivals.py):

```python
"myengine": Rival(
    key="myengine", name="My Engine", port=9001,
    install_hint="Start it on port 9001 before launching the arena.",
),
```

Then add one `<option value="myengine">` to the dropdown in `ui/index.html`. If the engine
speaks the OpenAI streaming API, nothing else is needed — subclass `Rival` and override
`available()` / `launch_cmd()` only if you want the arena to start it for you.

---

## Bring your own model

HIVE Limited quantizes at load time from HuggingFace safetensors, so point `serve.py` at any
directory with `config.json` + `*.safetensors`:

```bash
python3 serve.py /path/to/your-model
```

The prebuilt kernels in this build target the **Qwen3 architecture** (RMSNorm, RoPE, GQA,
SwiGLU, per-head q/k norm). Other architectures need different kernels, which this build does
not ship — that is one of its documented omissions, not a bug.

To race fairly, give the rival **the same weights**:

- **llama.cpp** wants one `.gguf` in `~/.hive-limited/models/` (it picks the first
  alphabetically). Convert your safetensors with llama.cpp's own `convert_hf_to_gguf.py`,
  or download a matching GGUF.
- **vLLM** wants a HuggingFace directory and picks the first one containing `config.json`.

Different sizes are fine — `MODEL_REPO=Qwen/Qwen3-4B ./setup/download_models.sh` — as long as
your card fits both engines one at a time.

---

## The hardware panel, if you run under WSL

The panel at the bottom of the arena reads your machine live. The **GPU rows are always
real** — the card is passed through, so `libcuda` reports the actual device.

The **CPU and RAM rows are not**, under WSL: a VM can only see its own allocation, which is
typically about half the machine, and interop to ask Windows is not always available. The
panel says so rather than presenting the VM's slice as the hardware.

If you know the machine's real specification, declare it:

```bash
export HIVE_LIMITED_HOST_CPU="Intel Core i9-10980HK @ 2.40GHz"
export HIVE_LIMITED_HOST_CORES="16 (8 physical cores)"
export HIVE_LIMITED_HOST_RAM_GIB="31.8"
python3 serve.py ~/.hive-limited/models/Qwen3-1.7B
```

Those rows then render marked **(declared)**, with a note explaining that they were supplied
rather than measured. Nothing typed there is ever presented as a measurement, and the
graphics section stays live regardless.

**On native Linux none of this applies** — `/proc` is the machine, and the panel is accurate
without any of these variables.

## Troubleshooting

**"not installed" next to an engine that is installed.** The arena looks in
`~/.hive-limited/engines/`, then on `PATH`. Either serve it on the port above, or point
`HIVE_LIMITED_ENGINES` at where it lives.

**A rival exits during startup.** The arena prints the exact command it used — run that by
hand and read the error. Model too large for VRAM is the usual cause.

**vLLM leaves GPU memory allocated.** It spawns a worker process that outlives a naive kill.
The arena launches it in its own process group and `/stop` kills the group; if you started it
yourself, stop it yourself.

**The race stalls at "waiting".** Only one heavy engine runs at a time by design. If a
previous run left something resident, press **STOP** — it frees every engine the arena
started.
