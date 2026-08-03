"""Rival engine adapters for the HIVE Limited arena.

You race HIVE Limited against engines **you** installed, on **your** machine. That is the
only kind of comparison worth anything: we ship the harness, you produce the result. No
figure in this repository is pre-recorded, including the rival's.

Both supported rivals (llama.cpp's `llama-server` and vLLM) expose an OpenAI-compatible
streaming API, so one client drives both. Written against `urllib` rather than `requests`
or the `openai` package so this repo keeps its numpy-only dependency promise.

WHAT THIS DELIBERATELY DOES NOT DO

  * it does not tune the rival. Default flags, documented in setup/, chosen to be ordinary
    rather than favourable to us. If you think a flag is unfair, change it -- it is your
    machine and the instructions say how.
  * it does not hide a rival's cold start. Model load time is reported as a `loading`
    event and is NOT counted in tokens/sec or first-token latency; both clocks start when
    the engine begins generating. Timing an engine's startup as if it were inference is
    the most common way these comparisons are rigged.
  * it does not fall back to a fake stream when an engine is missing. A missing engine
    reports how to install it and the lane stays empty.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("HIVE_LIMITED_HOME", Path.home() / ".hive-limited"))
ENGINES_DIR = Path(os.environ.get("HIVE_LIMITED_ENGINES", HOME / "engines"))
MODELS_DIR = Path(os.environ.get("HIVE_LIMITED_MODELS", HOME / "models"))


@dataclass
class Rival:
    """One rival engine: how to find it, how to start it, how to stream from it."""

    key: str
    name: str
    port: int
    install_hint: str
    #: set when the server is ours to manage; None means "already running, leave it alone"
    proc: subprocess.Popen | None = field(default=None, repr=False)
    #: The model to run, ASSIGNED by the server from the user's selection. A rival must
    #: never work this out for itself — see the note in launch_cmd below.
    model_name: str = ""
    model_dir: Path | None = None       # HuggingFace format (vLLM)
    model_file: Path | None = None      # GGUF (llama.cpp)
    #: True when the USER added this engine rather than it shipping with the build.
    custom: bool = False

    def use_model(self, model) -> None:
        """Point this rival at the model the user selected, before any launch."""
        if self.model_name and self.model_name != model.name and self.proc is not None:
            self.stop()                  # a running server holds the OLD model; drop it
        self.model_name = model.name
        self.model_dir = model.hf_dir
        self.model_file = model.gguf

    # ---- discovery ---------------------------------------------------------

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_up(self, timeout: float = 0.6) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(self.base_url() + path, timeout=timeout) as r:
                    if r.status == 200:
                        return True
            except Exception:
                continue
        return False

    def available(self) -> tuple[bool, str]:
        """(usable, explanation). Usable means installed OR already serving."""
        raise NotImplementedError

    def launch_cmd(self) -> list[str] | None:
        raise NotImplementedError

    # ---- lifecycle ---------------------------------------------------------

    def ensure(self, log, wait_s: int = 600):
        """Start the server if needed and wait for it to answer. Yields loading events."""
        if self.is_up():
            return
        cmd = self.launch_cmd()
        if cmd is None:
            raise RuntimeError(self.install_hint)
        log(f"starting {self.name}…")
        # own process group so stop() can kill the whole tree; vLLM in particular spawns a
        # separate worker that holds the GPU memory and outlives a naive kill of the parent.
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited during startup (code {self.proc.returncode}). "
                    f"Run it by hand to see why:\n  {' '.join(cmd)}")
            if self.is_up():
                return
            log(f"loading {self.name} model… {int(time.time()-t0)}s")
            time.sleep(2.0)
        raise RuntimeError(f"{self.name} did not become ready within {wait_s}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            # REAP IT. A killed child that is never waited on stays a zombie, and under
            # WSL's WDDM the GPU context is not released while it does — measured: no live
            # llama-server, yet 4.6 GB still held. The kill is not finished until the
            # process table entry is gone.
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
        self.proc = None
        self.stop_orphan()

    def stop_orphan(self) -> None:
        """Kill a leftover copy of an engine WE installed, from a previous server run.

        `self.proc` only knows about servers this process started. A rival left running
        when the arena restarts is invisible to it — so it keeps holding its VRAM (measured:
        ~5.5 GB of an 8 GB card), `is_up()` reports it as serving, and the next model load
        has nothing left to load into and appears to hang. That is exactly the "stuck at
        loading" failure.

        Only processes running the binary from OUR engines directory are killed. An engine
        the user started themselves is left alone — the docs promise it is used as-is, and
        killing someone's own server because it occupies a port we like would be worse than
        the problem.
        """
        marker = str(ENGINES_DIR)
        try:
            out = subprocess.run(["pgrep", "-af", marker],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return
        for line in out.splitlines():
            pid, _, cmd = line.partition(" ")
            if f"--port {self.port}" not in cmd and f"--port={self.port}" not in cmd:
                continue
            if marker not in cmd:
                continue                     # not ours; leave it be
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass

    # ---- streaming ---------------------------------------------------------

    def stream(self, prompt: str, ntok: int, model_id: str, stop_evt, log):
        """Yield {'tok': str, 't': ms-since-first-token} per token, OpenAI SSE.

        `t` is this process's clock, matching what the HIVE lane reports, so the two lanes
        are timed on one clock rather than two.
        """
        self.ensure(log)
        # RAW COMPLETION, not chat. This used to POST /v1/chat/completions, which makes the
        # server apply the model's chat template — and for a reasoning model like Qwen3 that
        # puts it into thinking mode, emitting a long internal monologue into a separate
        # `reasoning_content` field. HIVE's lane meanwhile just continues the text.
        #
        # So the two lanes were doing DIFFERENT TASKS: one continuing a sentence, the other
        # deliberating then answering. Every token counted on one side and none on the other
        # (the adapter only read `content`, which stayed empty). A race between different
        # tasks is meaningless however it is timed, so both lanes now do the same thing:
        # continue the prompt.
        body = {
            "model": model_id,
            "prompt": prompt,
            "stream": True,
            "temperature": 0.0,          # greedy on both sides, so the race is not a lottery
        }
        if ntok > 0:
            body["max_tokens"] = ntok
        req = urllib.request.Request(
            self.base_url() + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = None
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                if stop_evt.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                c0 = choices[0]
                # /v1/completions puts the text in `text`; the chat shape uses `delta`.
                # Both are read so an engine that only implements the chat endpoint still
                # works — and `reasoning_content` is included there, because dropping it
                # would silently discard every token a thinking model produces and report
                # the rival as having answered nothing.
                piece = c0.get("text") or ""
                if not piece:
                    delta = c0.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                if not piece:
                    continue
                now = time.perf_counter()
                if t0 is None:
                    t0 = now
                yield {"tok": piece, "t": round((now - t0) * 1e3, 3)}


class LlamaCpp(Rival):
    def available(self) -> tuple[bool, str]:
        if self.is_up():
            return True, "already serving"
        if self._binary():
            return True, "installed"
        return False, self.install_hint

    def _binary(self) -> str | None:
        local = ENGINES_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
        if local.exists():
            return str(local)
        return shutil.which("llama-server")

    def launch_cmd(self):
        exe = self._binary()
        if not exe:
            return None
        # The model is HANDED to us. This used to be `sorted(glob('*.gguf'))[0]`, which
        # agrees with the model HIVE is running only while exactly one is installed —
        # install a second and the two lanes silently diverge, producing a race between
        # different models that looks completely normal on screen.
        if not self.model_file:
            raise RuntimeError(
                f"'{self.model_name}' has no .gguf file, so llama.cpp cannot run it.\n\n"
                "HIVE reads HuggingFace files; llama.cpp reads GGUF. Both engines need "
                "their own copy of the SAME model for the race to mean anything.\n\n"
                "Get the missing one with:\n"
                "  ./setup/download_models.sh")
        return [exe, "-m", str(self.model_file), "--port", str(self.port),
                "-ngl", "99",                 # all layers on the GPU: their fast path
                "-c", "4096", "--no-warmup"]


class VLLM(Rival):
    def available(self) -> tuple[bool, str]:
        if self.is_up():
            return True, "already serving"
        if self._python():
            return True, "installed"
        return False, self.install_hint

    def _python(self) -> str | None:
        venv = ENGINES_DIR / "vllm-env" / "bin" / "python"
        if venv.exists():
            return str(venv)
        return shutil.which("vllm") and shutil.which("python3")

    def launch_cmd(self):
        py = ENGINES_DIR / "vllm-env" / "bin" / "python"
        if not py.exists():
            return None
        # Same rule as llama.cpp: the model is handed to us, never chosen here.
        if not self.model_dir:
            raise RuntimeError(
                f"'{self.model_name}' has no HuggingFace files, so vLLM cannot run it.\n\n"
                "Get them with:\n  ./setup/download_models.sh")
        return [str(py), "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(self.model_dir), "--port", str(self.port),
                "--gpu-memory-utilization", "0.85", "--max-model-len", "4096"]


class BringYourOwn(Rival):
    """An engine the arena never starts — you serve it, the arena races it.

    This is the honest way to include a stack we cannot install for you. TensorRT-LLM in
    particular needs a specific Python, CUDA libraries and MPI that no script here should
    be quietly putting on your machine. If it is answering on the port, it races; if it is
    not, the lane says exactly how to make it.
    """

    def available(self) -> tuple[bool, str]:
        return (True, "serving") if self.is_up() else (False, self.install_hint)

    def launch_cmd(self):
        return None


REGISTRY: dict[str, Rival] = {
    "trtllm": BringYourOwn(
        key="trtllm", name="TensorRT-LLM", port=8083,
        install_hint=("TensorRT-LLM is not running.\n\n"
                      "The arena does not install it for you: it needs a specific Python, "
                      "CUDA libraries and MPI, and putting those on your machine should be "
                      "your decision, not a side effect of a demo.\n\n"
                      "Serve it yourself with an OpenAI-compatible endpoint on port 8083:\n"
                      "  trtllm-serve <model> --port 8083\n\n"
                      "Then reload this page. See docs/ENGINES.md."),
    ),
    "llamacpp": LlamaCpp(
        key="llamacpp", name="llama.cpp", port=8081,
        install_hint=("llama.cpp is not installed. Install it with:\n"
                      "  ./setup/install_llamacpp.sh\n"
                      "It builds from source with CUDA and takes a few minutes."),
    ),
    "vllm": VLLM(
        key="vllm", name="vLLM", port=8082,
        install_hint=("vLLM is not installed. Install it with:\n"
                      "  ./setup/install_vllm.sh\n"
                      "It is a large download (several GB) into its own virtualenv."),
    ),
}


#: Engines the USER added, remembered between runs. The three above are the ones we happen
#: to have written installers for — they are not a statement about what is worth racing.
#: Anything speaking the OpenAI streaming API works, and this is how someone races a stack
#: we never thought about without editing any code.
CUSTOM_FILE = HOME / "custom_engines.json"


def _load_custom() -> None:
    try:
        data = json.loads(CUSTOM_FILE.read_text())
    except (OSError, ValueError):
        return
    for item in data if isinstance(data, list) else []:
        try:
            key = str(item["key"]); name = str(item["name"]); port = int(item["port"])
        except (KeyError, TypeError, ValueError):
            continue
        if key in REGISTRY:
            continue
        REGISTRY[key] = BringYourOwn(
            key=key, name=name, port=port, custom=True,
            install_hint=(f"{name} is not answering on port {port}.\n\n"
                          f"Start it yourself with an OpenAI-compatible endpoint on that "
                          f"port, then reload this page. The arena never launches an engine "
                          f"you added — it only uses one that is already running, exactly "
                          f"as you configured it."),
        )


def add_custom(name: str, port: int) -> tuple[bool, str]:
    """Register a user-supplied engine. Returns (ok, message)."""
    name = (name or "").strip()
    if not name:
        return False, "give the engine a name"
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False, "the port must be a number"
    if not (1 <= port <= 65535):
        return False, "the port must be between 1 and 65535"

    key = re.sub(r"[^a-z0-9]", "", name.lower())[:24] or f"custom{port}"
    if key in REGISTRY:
        return False, f"there is already an engine called {REGISTRY[key].name}"
    if any(r.port == port for r in REGISTRY.values()):
        clash = next(r.name for r in REGISTRY.values() if r.port == port)
        return False, f"port {port} is already used by {clash}"

    REGISTRY[key] = BringYourOwn(
        key=key, name=name, port=port, custom=True,
        install_hint=(f"{name} is not answering on port {port}.\n\n"
                      f"Start it yourself with an OpenAI-compatible endpoint on that port, "
                      f"then reload this page."),
    )
    saved = []
    for r in REGISTRY.values():
        if getattr(r, "custom", False):
            saved.append({"key": r.key, "name": r.name, "port": r.port})
    try:
        CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_FILE.write_text(json.dumps(saved, indent=2))
    except OSError as exc:
        return True, f"added, but could not be remembered for next time ({exc})"
    return True, f"{name} added on port {port}"


def remove_custom(key: str) -> bool:
    r = REGISTRY.get(key)
    if r is None or not getattr(r, "custom", False):
        return False
    r.stop()
    del REGISTRY[key]
    saved = [{"key": x.key, "name": x.name, "port": x.port}
             for x in REGISTRY.values() if getattr(x, "custom", False)]
    try:
        CUSTOM_FILE.write_text(json.dumps(saved, indent=2))
    except OSError:
        pass
    return True


def get(key: str) -> Rival | None:
    return REGISTRY.get(key)


def use_model(model) -> None:
    """Point every rival at the selected model. Called whenever the selection changes."""
    for r in REGISTRY.values():
        r.use_model(model)


def status(model=None) -> dict:
    """What the UI needs to tell the user which lanes will actually run.

    `can_run` is separate from `available` on purpose: an engine can be perfectly installed
    and still be unable to race the CURRENT model, because it has no file in the format it
    reads. Saying so before the race beats failing during it.
    """
    out = {}
    for key, r in REGISTRY.items():
        ok, why = r.available()
        can_run, why_not = True, None
        if model is not None:
            if key == "llamacpp" and not model.gguf:
                can_run, why_not = False, f"no GGUF file for {model.name}"
            elif key == "vllm" and not model.hf_dir:
                can_run, why_not = False, f"no HuggingFace files for {model.name}"
        out[key] = {"name": r.name, "available": ok, "serving": r.is_up(), "note": why,
                    "can_run": can_run, "why_not": why_not,
                    "custom": bool(getattr(r, "custom", False)), "port": r.port}
    return out


def stop_all() -> None:
    """Stop every rival we started — and sweep orphans from a previous server run.

    stop() already calls stop_orphan(), so a rival this process never launched is still
    cleared. That is the case that caused the hang: after a restart, `proc` is None for a
    llama-server still holding 5.5 GB, so nothing here could see it.
    """
    for r in REGISTRY.values():
        r.stop()


_load_custom()
