"""Tokenizer access for the demo server — one loaded instance, resolved once.

Thin by design: the tokenizer itself is qwen3_tokenizer.py, a standard-library
implementation of the published Qwen3 tokenizer.json pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

from qwen3_tokenizer import Qwen3NativeTokenizer

_INSTANCE = None
_MODEL_DIR = None


def configure(model_dir: str | Path) -> None:
    global _MODEL_DIR, _INSTANCE
    _MODEL_DIR, _INSTANCE = Path(model_dir), None


def get() -> Qwen3NativeTokenizer:
    global _INSTANCE
    if _INSTANCE is None:
        model_dir = _MODEL_DIR or os.environ.get("HIVE_LIMITED_MODEL")
        if model_dir is None:
            raise RuntimeError("tokenizer model directory not configured")
        _INSTANCE = Qwen3NativeTokenizer(model_dir)
    return _INSTANCE
