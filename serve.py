"""A local, single-user demo surface for HIVE Limited.

This is NOT a serving layer. It binds to loopback, runs one request at a time against a
batch-1 engine, and has no queue, no concurrency, no batching and no auth. It exists so you
can watch the engine work, race it against engines you installed, and check its receipts in
a browser rather than reading a terminal. Anything resembling production serving is absent
(CONTENTS.md).

Endpoints — the core set is fixed by the arena UI, which is a copy rather than a rewrite;
the rest exist for the setup page (models, engines, installs):

    GET  / and /setup.html                     the pages
    GET  /run?backend=&prompt=&ntok=           Server-Sent Events, one event per token
    GET  /engines                              which rivals are installed / serving
    GET  /engines/add, /engines/remove         connect / forget a custom OpenAI-compat rival
    GET  /models, /select                      installed model list; switch the loaded model
    GET  /install?what=, /installable          run an allow-listed setup script; what exists
    GET  /scoreboard                           board rows + the honest caveats under the race
    GET  /receipt                              which shared libraries are actually mapped
    GET|POST /stop                             halt the current race and free rival GPUs

Cross-site guard: the side-effect routes (/run, /stop, /select, /install, /engines/*)
refuse requests a modern browser marks `Sec-Fetch-Site: cross-site`, unless the
initiator is itself loopback (localhost vs 127.0.0.1 count as different sites, but a
loopback Origin/Referer is our own UI). A random web page you have open must not be
able to poke a local server that can start installs or kill races; page loads,
read-only endpoints, curl, and app-launched navigations are unaffected.

THE STREAM PROTOCOL (unnamed SSE `message` events, one JSON object each):

    {"loading": "..."}                 engine is cold-loading; not counted in any timing
    {"error":   "..."}                 backend failed; the lane shows this and stops
    {"tok": "text", "t": ms}           one token, `t` = ms since THIS lane's first token
    {"done": true, "stats": {...}}     finished

`t` comes from the server for both lanes, so the two engines are timed on one clock. The UI
records the offsets and replays the race at its real cadence; recording client arrival times
instead would time the transport rather than the engine.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

import models as model_lib  # noqa: E402  (after sys.path)
import rivals  # noqa: E402

ENGINE = None
MODEL_DIR = None
MODEL = None                        # the selected model_lib.Model, shared by ALL lanes
ENGINE_LOCK = threading.Lock()      # one heavy engine at a time, by construction
STOP = threading.Event()

#: Caveats shown under the race. These are about THIS build and this harness — they are the
#: things a reader would otherwise have to take on trust, written down where the numbers are.
SCOREBOARD_NOTES = [
    {"icon": "⚠", "title": "This is not HIVE",
     "short": "A deliberately limited build.",
     "body": "HIVE Limited is batch-1 only, ships one quantization form, and has no "
             "optimised prefill path. The batch and serving paths, the compiler toolchain "
             "and the kernel research record are absent, not disabled. Do not report these "
             "numbers as HIVE's. Dated full-engine results are in REFERENCE.md."},
    {"icon": "⚖", "title": "NOT the same bit-width",
     "short": "Check formats before speeds.",
     # The rival's actual format is filled in at request time — see scoreboard_notes().
     # It used to assert "llama.cpp Q4_K_M is roughly 5.0-5.3", which is simply wrong for
     # the default model: Qwen3-1.7B has no 4-bit GGUF published, only Q8_0 at ~8.5.
     # Asserting the fair case while the unfair one is what is installed is the exact
     # failure this note exists to prevent.
     "body": "HIVE Limited runs int4 with a per-32 scale = 5.00 bits/weight, measured from "
             "the weights it loaded. Fewer bits means fewer bytes read per token, and decode "
             "is memory-bound, so part of any margin is the FORMAT rather than the engine. "
             "Race a rival at a comparable bit-width if you want the engine-only answer."},
    {"icon": "⏱", "title": "What the clock includes",
     "short": "Model loading is never timed.",
     "body": "Both lanes are timed by this server on one clock. Cold model load is reported "
             "as a loading event and excluded from tokens/sec and first-token latency; the "
             "clocks start when an engine begins generating. On a small card the engines "
             "cannot co-reside, so they run one at a time and the UI replays them together "
             "afterwards -- the race you watch is a synchronised replay of two real runs."},
    {"icon": "⚙", "title": "Rivals run at their defaults",
     "short": "Your machine, your flags.",
     "body": "Rivals are launched with ordinary settings (llama.cpp with all layers on the "
             "GPU, vLLM with a standard memory fraction), chosen to be unremarkable rather "
             "than favourable to us, and both are run greedy so the race is not a lottery. "
             "They are listed in setup/. If you think a flag is unfair, change it -- this "
             "runs on your machine and the instructions say how."},
]


#: The comparison board — results from the **FULL** engine, which is not in this repository.
#:
#: This is the one place the demo shows a number it did not just measure, and it is fenced
#: accordingly. REFERENCE.md is canonical for these rows — when a row moves there, update
#: it here in the same commit (there is no automatic cross-check; the duty is manual and
#: this sentence is the reminder). Rules, identical to REFERENCE.md:
#:   * claim-grade only: same-round interleaved medians, gates counted, token-gated against
#:     an independent oracle. No projections, no "up to".
#:   * LOSSES beside wins. NVIDIA leads four rows here and they are stated plainly.
#:   * withdrawn results stay withdrawn — several earlier HIVE figures were retired when
#:     their measurement method proved faulty; none appear.
#:   * a row whose correctness label is still open SAYS SO in its basis line.
#:   * bit-width asymmetry disclosed wherever a pairing is not equal-bit-width.
FULL_ENGINE_BOARD = [
    # ---- H100 SXM5, Qwen3-1.7B vs TensorRT-LLM 1.2.1 W4A16-AWQ g128 -------------
    {"group": "h100", "axis": "Decode throughput, batch 1", "rival": "TensorRT-LLM 1.2.1",
     "ours": "594.8 tok/s", "theirs": "494.6 tok/s", "verdict": "OURS 1.20×",
     "basis": "H100 SXM5 · 3 interleaved rounds · certified per-32 lane, correctness label CLEAN · 2026-07-30"},
    {"group": "h100", "axis": "Decode throughput, batch 4", "rival": "TensorRT-LLM 1.2.1",
     "ours": "1765.2 tok/s", "theirs": "1562.6 tok/s", "verdict": "OURS 1.13×",
     "basis": "racing lane — correctness label still OPEN (one adjudication outstanding). "
              "The batch-1 win above does not depend on it. · 2026-07-30"},
    {"group": "h100", "axis": "Decode throughput, batch 8", "rival": "TensorRT-LLM 1.2.1",
     "ours": "2891.7 tok/s", "theirs": "2705.7 tok/s", "verdict": "OURS 1.07×",
     "basis": "racing lane — correctness label still OPEN, as above · 2026-07-30"},
    {"group": "h100", "axis": "Decode throughput, batch 16", "rival": "TensorRT-LLM 1.2.1",
     "ours": "3764.9 tok/s", "theirs": "4555.1 tok/s", "verdict": "THEIRS 1.21×",
     "basis": "the batch deficit on Hopper is real and is an open engineering front · 2026-07-30"},
    {"group": "h100", "axis": "Decode throughput, batch 32", "rival": "TensorRT-LLM 1.2.1",
     "ours": "4227.8 tok/s", "theirs": "8312.5 tok/s", "verdict": "THEIRS 1.97×",
     "basis": "measured 2026-07-30"},
    {"group": "h100", "axis": "Decode throughput, batch 64", "rival": "TensorRT-LLM 1.2.1",
     "ours": "≈4.7k tok/s", "theirs": "≈14.7k tok/s", "verdict": "THEIRS 3.1×",
     "basis": "measured 2026-08-02 · the widest deficit on the board, stated plainly"},
    {"group": "h100", "axis": "Time to first token, 512-token prompt", "rival": "TensorRT-LLM 1.2.1",
     "ours": "17.5 ms", "theirs": "6.23 ms", "verdict": "THEIRS 2.8×",
     "basis": "prefill on Hopper is NVIDIA's by a wide margin — stated, not buried · 2026-07-30"},

    # ---- RTX 3070 Laptop, Qwen3-4B, EQUAL int4 both sides -----------------------
    {"group": "vllm", "axis": "Prefill, 1024-token prompt", "rival": "vLLM",
     "ours": "206.6 ms", "theirs": "307.1 ms", "verdict": "OURS 1.49×",
     "basis": "Qwen3-4B, int4 BOTH sides — the fairest pairing on this board. 3 rounds, "
              "pairwise 3/3, palindrome order. 1.38× on the conservative pairing. · 2026-07-24"},
    {"group": "vllm", "axis": "Decode latency per token", "rival": "vLLM",
     "ours": "7.73 ms", "theirs": "10.13–14.22 ms", "verdict": "OURS 1.3–1.8×",
     "basis": "Qwen3-4B AWQ-int4, CUDA graphs on both sides · 2026-07-21"},

    # ---- RTX 3070 Laptop, Qwen3-8B vs llama.cpp --------------------------------
    {"group": "llama", "axis": "Prefill, 1024-token prompt", "rival": "llama.cpp",
     "ours": "338.7 ms", "theirs": "453.4 ms", "verdict": "OURS 1.34×",
     "basis": "Qwen3-8B · 20 rounds · 60/60 gates · NOT equal bit-width (see below) · 2026-07-20"},
    {"group": "llama", "axis": "Decode latency per token", "rival": "llama.cpp",
     "ours": "12.24 ms", "theirs": "15.44 ms", "verdict": "OURS 1.26×",
     "basis": "Qwen3-8B · 12 rounds · 36/36 gates · NOT equal bit-width · 2026-07-20"},
    {"group": "llama", "axis": "Prefill, 2048-token prompt", "rival": "llama.cpp",
     "ours": "743.9–814.2 ms", "theirs": "933.4–1969.7 ms", "verdict": "OURS 1.16–2.50×",
     "basis": "the spread is llama.cpp's, not ours — its f16 path degrades under sustained load · 2026-07-20"},

    # ---- structural ------------------------------------------------------------
    {"group": "struct", "axis": "8B int4 with CUDA graphs in 8 GB", "rival": "vLLM",
     "ours": "runs", "theirs": "out of memory", "verdict": "structural",
     "basis": "vLLM falls back to eager mode; a capability difference, not a speed result"},
    {"group": "struct", "axis": "What the stack requires to run", "rival": "every engine here",
     "ours": "GPU driver only", "theirs": "CUDA toolkit + cuBLAS/cuDNN, or PyTorch",
     "verdict": "structural",
     "basis": "the receipt panel above reads this process's mapped libraries so you can check it"},
    {"group": "struct", "axis": "Token-identical across GPU generations", "rival": "not offered",
     "ours": "sm_86 · sm_89 · sm_90", "theirs": "—", "verdict": "structural",
     "basis": "same compiler-emitted PTX, no per-architecture source; full identity battery on each"},
]


def free_vram_gib() -> float:
    try:
        import cudrv
        return cudrv.mem_info()[0] / 2**30
    except Exception:
        return -1.0


def load_engine(model_dir: str):
    """Load (or reload) the engine, and point every rival at the same model.

    The single most important line here is `rivals.use_model(MODEL)`: both lanes must be
    running the same model or the race is meaningless, and that has to be enforced in one
    place rather than trusted to each adapter.
    """
    global ENGINE, MODEL_DIR, MODEL
    import limited
    import tokenizer_min

    resolved = model_lib.find(model_dir)
    if resolved is None or resolved.hf_dir is None:
        raise FileNotFoundError(
            f"no HuggingFace model found at or named {model_dir!r} "
            f"(looked in {model_lib.MODELS_DIR})")

    if ENGINE is not None:                    # switching: free the old weights first
        try:
            ENGINE.free()
        except Exception:
            pass
        ENGINE = None

    # A rival left running by a previous session holds its VRAM and is invisible to this
    # process — that is what made loading appear to hang rather than fail. Clear ours out
    # first, and say plainly if the card is still too full to succeed, because "loading…"
    # forever is the least useful thing a program can do.
    rivals.stop_all()
    # free_vram_gib() returns -1 when it cannot read — before the first load there is no
    # CUDA context to ask. Report a figure only when there is one, rather than printing
    # "-1.0 GiB free", which reads like the card is broken.
    free = free_vram_gib()
    if 0 <= free < 1.5:
        print(f"WARNING: only {free:.1f} GiB of GPU memory is free. Something else is using "
              f"this card — loading may fail or be very slow.", flush=True)

    MODEL = resolved
    MODEL_DIR = str(resolved.hf_dir)
    tokenizer_min.configure(MODEL_DIR)
    where = f" ({free:.1f} GiB free)" if free >= 0 else ""
    print(f"loading {MODEL.name} from {MODEL_DIR} ...{where}", flush=True)
    ENGINE = limited.Engine(MODEL_DIR, kernels_dir=str(HERE / "kernels"))

    # The decode step is ~538 kernel launches at ~25 us of host time each, so issuing them
    # one at a time leaves the GPU idle ~82% of the step. Replaying them from a CUDA graph
    # removes that, measured 2.85x here — but ONLY if it is provably the same computation,
    # so the fast path is earned rather than assumed. A mismatch disables it and says so
    # instead of shipping an engine that is fast and wrong.
    try:
        ids = tokenizer_min.get().encode("The capital of France is")
        t0 = time.perf_counter()
        ENGINE._graph_ok = ENGINE.verify_graph(ids, n=24)
        took = time.perf_counter() - t0
        if ENGINE._graph_ok:
            print(f"graph replay verified identical over 24 tokens ({took:.1f}s) — enabled",
                  flush=True)
        else:
            print("graph replay DIVERGED from the direct path — disabled, running eager",
                  flush=True)
    except Exception as exc:
        ENGINE._graph_ok = False
        print(f"graph replay unavailable ({type(exc).__name__}) — running eager", flush=True)

    rivals.use_model(MODEL)
    print(f"ready on {ENGINE.device_name}", flush=True)
    return ENGINE


def mapped_libraries() -> dict:
    """Read this process's mapped shared objects.

    The point of showing it: you can confirm for yourself that no cuBLAS, cuDNN, TensorRT
    or CUDA-runtime library is loaded — only the driver. It is a check you run, not a claim
    we make.
    """
    interesting = ("cublas", "cudnn", "cufft", "cusparse", "cusolver", "nvrtc",
                   "tensorrt", "nvinfer", "cudart", "torch", "nccl")
    found, driver = [], []
    if os.name == "nt":
        # Native Windows: enumerate this process's loaded modules through psapi —
        # the same check /proc/self/maps gives on Linux. Before this branch existed
        # the panel showed "available: false" fields as a broken-looking receipt.
        import ctypes
        from ctypes import wintypes
        psapi = ctypes.windll.psapi
        k32 = ctypes.windll.kernel32
        # Explicit prototypes matter: without them ctypes marshals the -1 pseudo-handle
        # through a 32-bit int and the call fails with INVALID_HANDLE on 64-bit Python.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EnumProcessModulesEx.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        psapi.GetModuleFileNameExW.argtypes = [
            wintypes.HANDLE, wintypes.HMODULE, ctypes.c_wchar_p, wintypes.DWORD]
        handle = k32.GetCurrentProcess()
        arr = (wintypes.HMODULE * 2048)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(handle, arr, ctypes.sizeof(arr),
                                          ctypes.byref(needed), 0x03):
            return {"available": False, "note": "could not enumerate loaded modules"}
        count = min(needed.value // ctypes.sizeof(wintypes.HMODULE), 2048)
        buf = ctypes.create_unicode_buffer(1024)
        for i in range(count):
            if not psapi.GetModuleFileNameExW(handle, arr[i], buf, 1024):
                continue
            base = os.path.basename(buf.value)
            low = base.lower()
            if any(tag in low for tag in interesting):
                if base not in found:
                    found.append(base)
            elif low.startswith("nvcuda") and base not in driver:
                driver.append(base)
        return {
            "available": True,
            "driver": driver,
            "math_or_inference_libraries": found,
            "clean": len(found) == 0,
            "note": ("no NVIDIA math or inference library is mapped — only the driver"
                     if not found else
                     "unexpected libraries are mapped; this build should not need them"),
        }
    try:
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                path = line.rstrip().split(" ", 5)[-1].strip()
                if not path.startswith("/") or ".so" not in path:
                    continue
                base = os.path.basename(path)
                low = base.lower()
                if any(tag in low for tag in interesting):
                    if base not in found:
                        found.append(base)
                elif "libcuda.so" in low and base not in driver:
                    driver.append(base)
    except FileNotFoundError:
        return {"available": False,
                "note": "/proc/self/maps is Linux-only; run under Linux or WSL to see this"}
    return {
        "available": True,
        "driver": driver,
        "math_or_inference_libraries": found,
        "clean": len(found) == 0,
        "note": ("no NVIDIA math or inference library is mapped — only the driver"
                 if not found else
                 "unexpected libraries are mapped; this build should not need them"),
    }


#: Scripts the page is allowed to run, by name. An ALLOW-LIST, not a command parameter:
#: the endpoint takes a key from this table and nothing else, so no request can ask the
#: server to run something that is not written here. The server is loopback-only and
#: single-user, and clicking a button is no more privileged than typing the same line in
#: the terminal it was started from — but "no more privileged than" is only true while the
#: set of runnable things is fixed in advance.
INSTALLABLE = {
    "llamacpp": {"script": "setup/install_llamacpp.sh", "label": "llama.cpp",
                 "mins": "about 5 minutes"},
    "vllm": {"script": "setup/install_vllm.sh", "label": "vLLM",
             "mins": "about 10 minutes"},
    "model": {"script": "setup/download_models.sh", "label": "model",
              "mins": "a few minutes"},
}
INSTALL_LOCK = threading.Lock()      # one install at a time; they all compete for disk/GPU


def tilde(path) -> str:
    """Abbreviate the operator's home directory to `~` before anything reaches the browser.

    A full path like /home/alice/.hive-limited/models/Qwen3-1.7B answers a question nobody
    asked and prints the operator's USERNAME on a page this project actively encourages
    people to screenshot and share. `~/.hive-limited/models/Qwen3-1.7B` is more readable and
    carries no one's name.
    """
    if not path:
        return "—"
    text = str(path)
    home = str(Path.home())
    # Replace ANYWHERE, not just at the start. Streamed installer output embeds paths
    # mid-line ("Cloning into '/home/alice/...'", "Installed models in /home/alice/..."),
    # and a startswith-only version silently let every one of those through.
    return text.replace(home, "~") if home else text


def scoreboard_notes() -> list[dict]:
    """The caveat cards, with the bit-width one told what the rival is REALLY running.

    A fixed string here would keep asserting the fair case (Q4_K_M ~5 bpw) while an unfair
    one is installed — the default model publishes only a Q8_0 GGUF at ~8.5 bpw against
    HIVE's 5.00. A note whose whole job is to flag a mismatch must not be the thing hiding
    one.
    """
    notes = [dict(n) for n in SCOREBOARD_NOTES]
    if MODEL is not None and MODEL.gguf_quant:
        j = MODEL.as_json()
        bpw, quant = j.get("gguf_bpw"), j["gguf_quant"]
        if bpw:
            verdict = ("comparable — this is close to a like-for-like race"
                       if abs(bpw - 5.0) <= 0.9 else
                       f"NOT comparable — llama.cpp is reading about {bpw / 5.0:.1f}x "
                       f"the bytes per weight, so expect it to be slower for reasons that "
                       f"have nothing to do with its engine")
            for n in notes:
                if n["title"].startswith("NOT the same"):
                    n["body"] += (f"  Right now llama.cpp would run {MODEL.name} as "
                                  f"{quant} (~{bpw} bits/weight) against HIVE's 5.00: "
                                  f"{verdict}.")
    return notes


def model_json(model) -> dict | None:
    """A model as the BROWSER should see it: home-abbreviated, everywhere, one place."""
    if model is None:
        return None
    j = model.as_json()
    return {**j,
            "hf_dir": tilde(j["hf_dir"]) if j["hf_dir"] else None,
            "gguf": tilde(j["gguf"]) if j["gguf"] else None}


def rig_spec() -> list[dict]:
    """The machine THIS page is running on, read live.

    The internal build's version of this panel was a spec sheet for the laptop its recorded
    numbers came from. Here there are no recorded numbers, so a fixed spec sheet would be
    describing someone else's computer. Every value below is read at request time — the GPU
    fields through the same `libcuda` binding the engine uses, the rest from `/proc`.
    """
    groups: list[dict] = []

    gpu: list[list[str]] = []
    try:
        import cudrv
        # CUDA contexts are per-thread and this runs on an HTTP worker thread, so the
        # context-dependent calls (cuMemGetInfo) fail with INVALID_CONTEXT unless the thread
        # is bound first. The device queries above it happen to be context-free, which is
        # why the group came back with only its VRAM row missing rather than empty.
        if ENGINE is not None:
            ENGINE.bind_thread()
        dev = cudrv.device(0)

        def row(label, fn):
            """One field per try: a driver call that fails should cost its own row, not the panel."""
            try:
                gpu.append([label, fn()])
            except Exception as exc:
                gpu.append([label, f"unavailable ({type(exc).__name__})"])

        # 16 = MULTIPROCESSOR_COUNT, 36 = MEMORY_CLOCK_RATE (kHz), 37 = GLOBAL_MEMORY_BUS_WIDTH
        row("GPU", lambda: cudrv.device_name(dev))
        row("compute capability", lambda: "sm_%d%d" % cudrv.compute_capability(dev))
        row("streaming multiprocessors", lambda: str(cudrv.device_attribute(dev, 16)))
        row("VRAM", lambda: "%.1f GiB  (%.1f GiB free)" % tuple(
            x / 2**30 for x in reversed(cudrv.mem_info())))
        row("peak memory bandwidth", lambda: "%.0f GB/s (theoretical)" % (
            2 * cudrv.device_attribute(dev, 36) * (cudrv.device_attribute(dev, 37) / 8) / 1e6))
        row("driver (CUDA version)", cudrv.driver_version)
    except Exception as exc:
        gpu = [["GPU", f"could not read: {exc}"]]
    groups.append({"g": "Graphics — read through libcuda, the same binding the engine uses",
                   "rows": gpu})

    # Under WSL, /proc describes the VM's slice, NOT the machine. On the development laptop
    # that is 21.5 GiB of a real 31.8, and 8 logical cores of a real 16 — so a panel headed
    # "Host" would understate the hardware by a third while looking authoritative. The GPU
    # rows above are unaffected: the card is passed through, so libcuda reports the real
    # device. Label what is actually being measured rather than guessing at the host.
    wsl = False
    try:
        with open("/proc/version") as fh:
            wsl = "microsoft" in fh.read().lower()
    except OSError:
        pass

    env_cpu = os.environ.get("HIVE_LIMITED_HOST_CPU")
    env_cores = os.environ.get("HIVE_LIMITED_HOST_CORES")
    env_ram = os.environ.get("HIVE_LIMITED_HOST_RAM_GIB")
    declared = any((env_cpu, env_cores, env_ram))

    host: list[list[str]] = []
    try:
        model = None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        cores = os.cpu_count()
        mem_kb = None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    break
        # A VM cannot see past its own allocation, and WSL interop is not always available
        # to ask the host, so the operator may DECLARE the real figures (env vars read
        # above). Declared values are labelled as declared — this panel must never present
        # something typed as something measured. On native Linux /proc is the machine and
        # none of this applies.
        def mark(measured: bool) -> str:
            return "" if measured else " (declared)"

        rows = []
        if env_cpu or model:
            rows.append(["CPU" + mark(not env_cpu), env_cpu or model])
        if env_cores or cores:
            rows.append(["logical cores" + mark(not env_cores),
                         env_cores or str(cores)])
        if env_ram or mem_kb:
            rows.append(["RAM" + mark(not env_ram),
                         f"{float(env_ram):.1f} GiB" if env_ram
                         else f"{mem_kb / 2**20:.1f} GiB"])
        if hasattr(os, "uname"):
            rows.append(["kernel", " ".join(os.uname()[:3])])

        if wsl and not declared:
            rows.append(["note", "running under WSL — the CPU and RAM above are this VM's "
                                 "allocation, not the machine's. The GPU is passed through, "
                                 "so the graphics figures are the real device. Set "
                                 "HIVE_LIMITED_HOST_CPU / _CORES / _RAM_GIB to show the "
                                 "machine's real specification."])
        elif wsl and declared:
            rows.append(["note", "running under WSL, which can only see its own allocation, "
                                 "so the rows marked (declared) were supplied by the operator "
                                 "rather than measured. Everything unmarked — and the whole "
                                 "graphics section — is read live."])
        host = rows
    except Exception as exc:
        host = [["host", f"could not read: {exc}"]]
    title = "Host"
    if wsl:
        title = "Host — declared where WSL cannot see it" if declared else \
                "Host — as seen by this process"
    groups.append({"g": title, "rows": host})

    # The model's NAME, not its path. This row used to print the full filesystem location,
    # which was a leftover from before models had names — it answered a question nobody asked
    # and put the operator's username in every screenshot of this panel.
    model_row = "—"
    if MODEL is not None:
        params = (MODEL.catalogue or {}).get("params")
        model_row = MODEL.name + (f" · {params}" if params else "")
    stack = [
        ["engine", "HIVE Limited — batch-1, prebuilt PTX"],
        ["python", sys.version.split()[0]],
        ["model", model_row],
    ]
    if ENGINE is not None:
        stack += [
            ["quantization", f"int4, per-32 scale — {ENGINE.bits_per_weight():.2f} bits/weight"],
            ["weights read per token", f"{ENGINE.weight_bytes() / 1e9:.3f} GB"],
        ]
    try:
        import numpy
        stack.append(["numpy", numpy.__version__])
    except Exception:
        pass
    groups.append({"g": "This build", "rows": stack})
    return groups


def our_gb_per_token() -> float | None:
    """Bytes read per token, from the weights actually loaded — not a constant."""
    if ENGINE is None:
        return None
    try:
        return round(ENGINE.weight_bytes() / 1e9, 4)
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # keep the console readable
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj, indent=2).encode())

    #: Routes that DO something (start a run, kill engines, launch installs). Only
    #: these carry the cross-site guard: page loads and read-only endpoints are
    #: harmless, and a foreign page cannot read their responses anyway (no CORS).
    SENSITIVE = {"/run", "/stop", "/select", "/install", "/engines/add", "/engines/remove"}

    def _cross_site(self, route):
        # A foreign web page's fetch()/img/script against this server carries
        # Sec-Fetch-Site: cross-site in every modern browser — refuse it for the
        # side-effect routes. Two look-alikes must stay ALLOWED: (a) navigations
        # launched by another app (cmd's `start`, the installer) are also marked
        # cross-site by Chromium, but they only ever open the page routes, which are
        # not guarded; (b) localhost and 127.0.0.1 are technically different sites,
        # so a page on one fetching the other is "cross-site" — if the INITIATOR is
        # itself loopback (Origin/Referer host), it is our own UI, not an attacker.
        if route not in self.SENSITIVE:
            return False
        if self.headers.get("Sec-Fetch-Site", "").lower() != "cross-site":
            return False
        src = self.headers.get("Origin") or self.headers.get("Referer") or ""
        host = urlparse(src).hostname or ""
        if host in ("localhost", "127.0.0.1", "::1"):
            return False
        self._send(403, "text/plain; charset=utf-8",
                   b"cross-site requests to this local server are refused")
        return True

    def do_POST(self):
        if self._cross_site(urlparse(self.path).path):
            return
        if urlparse(self.path).path == "/stop":
            return self._stop()
        return self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_GET(self):
        url = urlparse(self.path)
        route = url.path
        if self._cross_site(route):
            return
        if route in ("/", "/index.html", "/setup.html"):
            name = "setup.html" if route == "/setup.html" else "index.html"
            page = HERE / "ui" / name
            if not page.exists():
                return self._send(404, "text/plain; charset=utf-8",
                                  f"ui/{name} not found".encode())
            return self._send(200, "text/html; charset=utf-8", page.read_bytes())
        if route == "/receipt":
            return self._json(mapped_libraries())
        if route == "/scoreboard":
            # `measured` carries FULL-ENGINE results — the one place this demo shows a
            # figure it did not just measure. The board's own header says so, and every
            # row is claim-grade with its losses and open labels stated. See the comment
            # on FULL_ENGINE_BOARD for the rules it is held to.
            return self._json({"notes": scoreboard_notes(), "measured": FULL_ENGINE_BOARD,
                               "rig": rig_spec()})
        if route == "/engines":
            return self._json({
                "ours": {"name": "HIVE Limited",
                         "device": getattr(ENGINE, "device_name", None),
                         "bits_per_weight": (round(ENGINE.bits_per_weight(), 3)
                                             if ENGINE else None),
                         "gb_per_token": our_gb_per_token()},
                "rivals": rivals.status(MODEL),
                "model": model_json(MODEL),
            })
        if route == "/models":
            # Installed models + the download menu, so the page can offer both "switch to
            # one you have" and "get one you don't" without the user touching a path.
            # Paths are home-abbreviated: they are here to help someone find their own
            # folder, not to publish the account name they are logged in as.
            return self._json({
                "installed": [model_json(m) for m in model_lib.discover()],
                "current": MODEL.key if MODEL else None,
                "catalogue": model_lib.catalogue_json(),
                "models_dir": tilde(model_lib.MODELS_DIR),
            })
        if route == "/select":
            return self._select(parse_qs(url.query))
        if route == "/engines/add":
            p = parse_qs(url.query)
            ok, msg = rivals.add_custom((p.get("name") or [""])[0],
                                        (p.get("port") or ["0"])[0])
            if ok and MODEL is not None:
                rivals.use_model(MODEL)       # a new engine must know the current model
            return self._json({"ok": ok, "message": msg}, 200 if ok else 400)
        if route == "/engines/remove":
            key = (parse_qs(url.query).get("key") or [""])[0]
            return self._json({"ok": rivals.remove_custom(key)})
        if route == "/install":
            return self._install(parse_qs(url.query))
        if route == "/installable":
            return self._json({k: {"label": v["label"], "mins": v["mins"]}
                               for k, v in INSTALLABLE.items()})
        if route == "/stop":
            return self._stop()
        if route == "/run":
            return self._run(parse_qs(url.query))
        return self._send(404, "text/plain; charset=utf-8", b"not found")

    def _stop(self):
        STOP.set()
        rivals.stop_all()
        return self._json({"stopped": True})

    def _install(self, params):
        """Run one allow-listed setup script, streaming its output to the page.

        This exists because the person reading the setup page is, by definition, already
        running the server — so telling them to open a terminal and retype a command the
        server could run itself is friction with nothing behind it.

        Output is streamed live rather than reported at the end: these take minutes, and a
        button that sits silent for five minutes is indistinguishable from one that is
        broken. The scripts already narrate what they are doing; this just carries it.
        """
        what = (params.get("what") or [""])[0]
        spec = INSTALLABLE.get(what)
        if spec is None:
            return self._json({"error": f"nothing installable called {what!r}"}, 404)

        script = HERE / spec["script"]
        if not script.exists():
            return self._json({"error": f"{spec['script']} is missing from this install"}, 500)

        args = []
        if what == "model":
            name = (params.get("name") or [""])[0]
            if not name:
                return self._json({"error": "which model?"}, 400)
            args = [name]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        if not INSTALL_LOCK.acquire(blocking=False):
            emit({"line": "Another install is already running. Wait for it to finish."})
            emit({"done": True, "ok": False})
            return

        try:
            emit({"line": f"Installing {spec['label']} — this takes {spec['mins']}."})
            emit({"line": ""})
            env = dict(os.environ)
            # The scripts prompt when run by hand. A prompt here would hang forever with
            # nobody to answer it, so they are told to fail with their explanation instead.
            env["HIVE_NONINTERACTIVE"] = "1"
            proc = subprocess.Popen(
                ["bash", str(script), *args],
                cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            gone = False
            for line in proc.stdout:
                if gone:
                    continue         # keep draining: see the note below
                try:
                    # Home-abbreviate the script's own output too. `git clone` prints the
                    # destination, which would otherwise put the operator's username into a
                    # log this page displays.
                    emit({"line": tilde(line.rstrip("\n"))})
                except ConnectionError:
                    # The reader navigated away. Deliberately DON'T kill the build — losing a
                    # five-minute compile to a stray tab close would be its own bug. But keep
                    # consuming stdout to the end, both so the pipe never fills and blocks the
                    # child, and so INSTALL_LOCK is held for the job's real duration rather
                    # than released while it is still running.
                    gone = True
            code = proc.wait()
            if gone:
                return
            emit({"line": ""})
            if code == 0:
                emit({"line": f"✓ {spec['label']} installed."})
            else:
                emit({"line": f"✗ Stopped (exit code {code}). The reason is above."})
            emit({"done": True, "ok": code == 0})
        except ConnectionError:
            pass                                  # user navigated away; the script keeps going
        except Exception as exc:
            try:
                emit({"line": f"✗ {type(exc).__name__}: {exc}"})
                emit({"done": True, "ok": False})
            except Exception:
                pass
        finally:
            INSTALL_LOCK.release()

    def _select(self, params):
        """Switch the model every lane uses. Reloads the engine; takes ~a minute."""
        key = (params.get("model") or [""])[0]
        target = model_lib.find(key)
        if target is None:
            return self._json({"ok": False,
                               "error": f"no model named {key!r} is installed"}, 404)
        if target.hf_dir is None:
            return self._json({"ok": False,
                               "error": f"{target.name} has no HuggingFace files, so HIVE "
                                        f"Limited cannot load it"}, 400)
        if MODEL is not None and target.key == MODEL.key:
            return self._json({"ok": True, "model": model_json(MODEL), "reloaded": False})
        # Serialise against a running generation: switching frees the current weights, and
        # doing that underneath an in-flight request would fault the GPU.
        with ENGINE_LOCK:
            rivals.stop_all()               # rivals hold the OLD model; they must restart
            try:
                load_engine(str(target.hf_dir))
            except Exception as exc:
                return self._json({"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._json({"ok": True, "model": model_json(MODEL), "reloaded": True})

    # ---- the race ----------------------------------------------------------

    def _run(self, params):
        # NO default backend. parse_qs drops blank values, so a fresh install whose
        # rival dropdown holds only "no rival engines installed yet" (value "") used to
        # fall through a `or ["ours"]` default here — the "rival" lane silently re-ran
        # our own engine and the banner declared a fake rival victory. An engine race
        # that isn't explicit does not run.
        backend = (params.get("backend") or [""])[0]
        prompt = (params.get("prompt") or [""])[0]
        raw_n = (params.get("ntok") or params.get("max_new") or ["100"])[0]
        try:
            ntok = int(raw_n)
        except ValueError:
            ntok = 100
        # -1 means "run to EOS". The real ceiling is applied per lane below: our engine
        # gets the model's remaining context, a rival gets its own launch window.
        unlimited = ntok < 0

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        STOP.clear()
        try:
            if not backend:
                emit({"error": "no engine selected for this lane — the dropdown has no "
                               "installed rival yet. ADD AN ENGINE installs or connects one."})
            elif backend == "ours":
                self._run_ours(prompt, ntok, emit, unlimited)
            else:
                self._run_rival(backend, prompt, ntok, emit, unlimited)
        except ConnectionError:
            pass
        except Exception as exc:
            try:
                emit({"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _run_ours(self, prompt, ntok, emit, unlimited=False):
        if ENGINE is None:
            return emit({"error": "engine not loaded"})
        import tokenizer_min
        tok = tokenizer_min.get()
        # The model's own chat template (non-thinking form). Raw encoding fed an
        # instruct model bare text, and greedy decoding of that produced degenerate
        # loops — "10000 words. 1. 10000 words. 2. ..." on the first real install.
        # bench/verify keep raw encoding: their golden outputs are pinned to it.
        ids = tok.prompt_ids(prompt)
        if unlimited:
            # "Unlimited" = to EOS or the model's real context edge — it used to be a
            # silent 512 cap, which is neither.
            ntok = max(1, ENGINE.cfg.max_context - len(ids) - 4)
        with ENGINE_LOCK:
            ENGINE.bind_thread()               # CUDA contexts are per-thread
            n, ttft_ms, prefill_ms = 0, None, None
            for item in ENGINE.stream(ids, max_new=ntok, stop_evt=STOP):
                if item[0] == "prefill":
                    prefill_ms = round(item[1] * 1e3, 2)
                    continue
                _, tid, off_s = item
                if ttft_ms is None:
                    ttft_ms = prefill_ms       # first token arrives after prefill
                n += 1
                emit({"tok": tok.decode([tid]), "t": round(off_s * 1e3, 3)})
            emit({"done": True, "stats": {
                "ttft_ms": ttft_ms,
                "tokens": n,
                "prefill_ms": prefill_ms,
                "bits_per_weight": round(ENGINE.bits_per_weight(), 3),
                # truthy => the UI shows the receipt panel, which then re-reads /receipt
                "receipt": True,
            }})

    def _run_rival(self, key, prompt, ntok, emit, unlimited=False):
        rv = rivals.get(key)
        if rv is None:
            return emit({"error": f"unknown backend {key!r}"})
        ok, why = rv.available()
        if not ok:
            return emit({"error": why})
        # The SAME templated string our lane encodes — one definition in the tokenizer
        # module — so both engines answer word-for-word identical input.
        import qwen3_tokenizer
        prompt = qwen3_tokenizer.chat_wrap(prompt)
        if unlimited:
            ntok = 3584                        # inside the rival's 4096-token launch window
        model_id = os.path.basename(str(MODEL_DIR or "model"))
        with ENGINE_LOCK:                      # never two heavy engines at once
            n, t_first = 0, None
            try:
                for chunk in rv.stream(prompt, ntok, model_id, STOP,
                                       lambda m: emit({"loading": m})):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
                    emit(chunk)
            finally:
                # RELEASE THE RIVAL. The race is record-then-replay, so once its tokens are
                # recorded its process is dead weight — and on a small card it is holding
                # most of the VRAM. Left running it starved the next model load until it
                # appeared to hang, which is the bug this fixes. An engine the user started
                # themselves is untouched; only one we launched is stopped.
                if rv.proc is not None:
                    rv.stop()
            emit({"done": True, "stats": {"ttft_ms": None, "tokens": n, "receipt": False}})


def main(argv):
    if len(argv) < 2:
        print("usage: python serve.py <model_dir> [port]")
        return 2
    port = int(argv[2]) if len(argv) > 2 else 8080
    load_engine(argv[1])

    class QuietServer(ThreadingHTTPServer):
        """A client closing its tab mid-stream is routine for SSE, not an error.

        The /run handler already catches BrokenPipe, but http.server ALSO flushes the
        socket in its own teardown after the handler returns — and that flush prints a
        full traceback to the terminal a beginner is watching. Same event, scarier
        clothes. ConnectionError covers the whole family - BrokenPipe, reset,
        and the ConnectionAborted form Windows raises for the same tab-close
        (WinError 10053, seen the first time the demo ran natively). Every
        other error still prints."""

        def handle_error(self, request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, ConnectionError):
                return
            super().handle_error(request, client_address)

    srv = QuietServer(("127.0.0.1", port), Handler)
    print(f"HIVE Limited arena on http://localhost:{port}  (loopback only)", flush=True)
    st = rivals.status()
    for key, s in st.items():
        mark = "ready" if s["available"] else "not installed"
        print(f"  rival {s['name']:<12} {mark}", flush=True)
    # Launchers set HIVE_OPEN_BROWSER=1 so the page opens exactly when the server can
    # answer it — a timer in a launcher lands on a dead page while the model is still
    # quantizing (found by the first real Windows install). The socket is already
    # bound and listening here; requests queue until serve_forever picks them up.
    if os.environ.get("HIVE_OPEN_BROWSER") == "1":
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        rivals.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
