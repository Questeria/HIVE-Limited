"""The model library: what is on disk, and which files belong to the same model.

Why this exists as its own module rather than a path argument:

A race is only meaningful if both lanes run the SAME model. Before this, HIVE ran whatever
path the server was started with while llama.cpp took the alphabetically-first `.gguf` it
could find and vLLM took the first directory containing a `config.json`. With one model
installed those agree by accident. With two they silently stop agreeing — you could watch
HIVE on a 4B model race llama.cpp on a 0.6B one, and the page would look entirely normal
while reporting a number that means nothing.

So models are discovered as **pairs**, and a rival is never allowed to pick for itself.

A model here is one logical thing (say "Qwen3-1.7B") with up to two files on disk:

  * a HuggingFace directory (`config.json` + `*.safetensors`) — read by HIVE Limited, which
    quantizes it at load time, and by vLLM
  * a GGUF file — read by llama.cpp

They are matched by a normalised name, so `Qwen3-1.7B/` pairs with
`Qwen3-1.7B-Q4_K_M.gguf`. A model with only one of the two still appears, and says which
engines can run it, because "you have the file for the wrong engine" is a far more useful
message than an empty dropdown.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path(os.environ.get("HIVE_LIMITED_MODELS",
                                 Path.home() / ".hive-limited" / "models"))

#: Curated, known-good models. Everything here is Qwen3, because the prebuilt kernels in
#: this build target that architecture — offering a Llama or Mistral here would be offering
#: a download that cannot run, which is worse than not offering it.
#:
#: `tested` names the published comparison each model backs, so a reader looking at a row on
#: the arena board can install the exact model behind it. What that buys is running the same
#: model against the same rival on their own hardware — NOT reproducing our figures, which
#: came from the full engine. Nothing in this repository can reproduce those, and the board
#: says so; pretending otherwise would be the one dishonest thing this project cannot afford.
CATALOGUE = [
    {"key": "qwen3-0.6b", "name": "Qwen3-0.6B", "params": "0.6B",
     "hf": "Qwen/Qwen3-0.6B", "gguf_repo": "Qwen/Qwen3-0.6B-GGUF",
     "gguf_file": "Qwen3-0.6B-Q4_K_M.gguf", "size": "~1.5 GB",
     "note": "Smallest and fastest. Good for a first run or a modest GPU.",
     "vram": "4 GB card is plenty"},
    {"key": "qwen3-1.7b", "name": "Qwen3-1.7B", "params": "1.7B",
     "hf": "Qwen/Qwen3-1.7B", "gguf_repo": "Qwen/Qwen3-1.7B-GGUF",
     "gguf_file": "Qwen3-1.7B-Q4_K_M.gguf", "size": "~4 GB",
     "note": "The default. Comfortable on an 8 GB card.", "default": True,
     "vram": "8 GB card", "tested": "the H100 rows — every batch size vs TensorRT-LLM"},
    {"key": "qwen3-4b", "name": "Qwen3-4B", "params": "4B",
     "hf": "Qwen/Qwen3-4B", "gguf_repo": "Qwen/Qwen3-4B-GGUF",
     "gguf_file": "Qwen3-4B-Q4_K_M.gguf", "size": "~9 GB",
     "note": "Better answers. Wants a 12 GB card or more to race comfortably.",
     "vram": "12 GB card", "tested": "the vLLM rows — the equal-int4 comparison"},
    {"key": "qwen3-8b", "name": "Qwen3-8B", "params": "8B",
     "hf": "Qwen/Qwen3-8B", "gguf_repo": "Qwen/Qwen3-8B-GGUF",
     "gguf_file": "Qwen3-8B-Q4_K_M.gguf", "size": "~17 GB",
     "note": "The largest we publish results for. Big download; quantizes on load, so it "
             "also wants ~20 GB of system RAM while it starts.",
     "vram": "8 GB card (tight) or more", "tested": "the llama.cpp rows"},
]


def _norm(text: str) -> str:
    """Fold a name so a directory and its GGUF match: 'Qwen3-1.7B' == 'qwen3_1_7b'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class Model:
    name: str
    hf_dir: Path | None = None
    gguf: Path | None = None
    catalogue: dict | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return _norm(self.name)

    @property
    def gguf_quant(self) -> str | None:
        """The GGUF's quantization, read from its filename ('Q8_0', 'Q4_K_M', …).

        This is not cosmetic. The official Qwen3 GGUF repos do not all publish a 4-bit
        build — the 0.6B and 1.7B ones ship only Q8_0, which is roughly 8.5 bits/weight
        against HIVE's 5.00. Racing those is not like-for-like, and the page has to be able
        to say WHICH format the rival is actually running rather than assuming Q4_K_M.
        """
        if not self.gguf:
            return None
        m = re.search(r"[-_.](Q\d[^.]*|BF16|F16|F32)\.gguf$", self.gguf.name, re.I)
        return m.group(1).upper() if m else None

    @property
    def engines(self) -> list[str]:
        """Which lanes can actually run this — used to warn before a race, not after."""
        out = []
        if self.hf_dir:
            out += ["hive", "vllm"]
        if self.gguf:
            out.append("llamacpp")
        return out

    def as_json(self) -> dict:
        c = self.catalogue or {}
        return {
            "key": self.key,
            "name": self.name,
            "hf_dir": str(self.hf_dir) if self.hf_dir else None,
            "gguf": str(self.gguf) if self.gguf else None,
            "engines": self.engines,
            "params": c.get("params"),
            "note": c.get("note"),
            "gguf_quant": self.gguf_quant,
            # Roughly how many bits/weight the rival's file carries, so the page can say
            # whether a race is like-for-like instead of assuming it is.
            "gguf_bpw": {"Q4_K_M": 4.8, "Q4_K_S": 4.6, "Q4_0": 4.5, "Q5_K_M": 5.7,
                         "Q5_0": 5.5, "Q6_K": 6.6, "Q8_0": 8.5,
                         "F16": 16.0, "BF16": 16.0, "F32": 32.0}.get(self.gguf_quant or ""),
            "tested": c.get("tested"),
            "vram": c.get("vram"),
            # The honest bit: say when a race would NOT be like-for-like.
            "can_race": bool(self.hf_dir and self.gguf),
            # LEAD WITH WHAT IS FINE. Two earlier attempts at this line both read as
            # "{name} is not downloaded" — the reader's eye lands on the model name next to
            # the word "not", and concludes the model they just installed is missing. It is
            # not: it is installed and working. What is absent is a second file that only
            # the OTHER engine reads, and that only matters if they want to race that engine.
            "missing": (None if (self.hf_dir and self.gguf)
                        else (f"{self.name} is installed and HIVE runs it. Racing llama.cpp "
                              f"as well needs its own copy in GGUF format — the same AI, "
                              f"packaged for that engine."
                              if self.hf_dir else
                              f"{self.name} is installed in GGUF format, which llama.cpp "
                              f"reads. HIVE and vLLM need the HuggingFace copy of it.")),
            "missing_short": (None if (self.hf_dir and self.gguf)
                              else ("HIVE only" if self.hf_dir else "llama.cpp only")),
            #: What to download to complete the pair, when we know how to get it.
            "completes_with": (None if (self.hf_dir and self.gguf) or not c.get("gguf_file")
                               else c.get("key")),
        }


def discover(models_dir: Path | None = None) -> list[Model]:
    """Everything installed, paired up. Never raises — an unreadable dir is simply empty."""
    root = Path(models_dir or MODELS_DIR)
    found: dict[str, Model] = {}
    if not root.is_dir():
        return []

    for entry in sorted(root.iterdir()):
        try:
            if entry.is_dir() and (entry / "config.json").exists():
                k = _norm(entry.name)
                found.setdefault(k, Model(name=entry.name))
                found[k].hf_dir = entry
                if not found[k].name:
                    found[k].name = entry.name
            elif entry.is_file() and entry.suffix == ".gguf":
                # 'Qwen3-1.7B-Q4_K_M.gguf' -> 'Qwen3-1.7B': drop the quantisation suffix so
                # it folds onto the directory of the same model.
                stem = re.sub(r"[-_.](Q\d[^.]*|f16|f32|bf16)$", "", entry.stem,
                              flags=re.I)
                k = _norm(stem)
                found.setdefault(k, Model(name=stem))
                found[k].gguf = entry
        except OSError:
            continue

    for m in found.values():
        for c in CATALOGUE:
            if _norm(c["name"]) == m.key:
                m.catalogue = c
                m.name = c["name"]
                break
    return sorted(found.values(), key=lambda m: m.name.lower())


def find(key_or_path: str, models_dir: Path | None = None) -> Model | None:
    """Resolve a selection: a library key, a model name, or an explicit path."""
    if not key_or_path:
        return None
    p = Path(key_or_path)
    if p.is_dir() and (p / "config.json").exists():
        # An explicit path given on the command line. Look for a GGUF beside it so a
        # hand-placed model still races rather than silently becoming HIVE-only.
        lib = {m.key: m for m in discover(models_dir)}
        match = lib.get(_norm(p.name))
        return Model(name=p.name, hf_dir=p,
                     gguf=match.gguf if match else None,
                     catalogue=match.catalogue if match else None)
    want = _norm(key_or_path)
    for m in discover(models_dir):
        if m.key == want:
            return m
    return None


def catalogue_json(models_dir: Path | None = None) -> list[dict]:
    """The download menu: every curated model, marked with whether it is already here."""
    have = {m.key for m in discover(models_dir)}
    out = []
    for c in CATALOGUE:
        out.append({**{k: v for k, v in c.items() if k != "default"},
                    "installed": _norm(c["name"]) in have,
                    "default": bool(c.get("default"))})
    return out
