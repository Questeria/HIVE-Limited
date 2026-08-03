"""Standard-library Qwen3 byte-level BPE tokenizer.

This implements the exact tokenizer.json pipeline used by Qwen3: NFC
normalization, the published Unicode-aware split pattern, GPT-2 byte mapping,
ranked BPE merges, added special tokens, and ByteLevel decoding. The single-user
chat shape is rendered directly instead of evaluating a third-party Jinja
template, which is why this needs no dependencies beyond the standard library.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Sequence


class Qwen3TokenizerError(RuntimeError):
    """Tokenizer artifacts or an encode/decode operation violated the contract."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHAT_PREFIX = "<|im_start|>user\n"
_CHAT_MIDDLE = "<|im_end|>\n<|im_start|>assistant\n"
_NO_THINK = "<think>\n\n</think>\n\n"
_REQUIRED_TEMPLATE_FRAGMENTS = (
    "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n'",
    "'<|im_start|>assistant\\n'",
    "'<think>\\n\\n</think>\\n\\n'",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen3TokenizerError(f"duplicate JSON key in tokenizer artifact: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], str, int]:
    if path.is_symlink() or not path.is_file():
        raise Qwen3TokenizerError(f"tokenizer artifact must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Qwen3TokenizerError(f"non-finite JSON value: {token}")
            ),
        )
    except Qwen3TokenizerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3TokenizerError(f"invalid tokenizer JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Qwen3TokenizerError(f"tokenizer artifact is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    byte_values = list(visible)
    codepoints = list(visible)
    extra = 0
    for value in range(256):
        if value not in visible:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    encode = dict(zip(byte_values, map(chr, codepoints), strict=True))
    return encode, {character: value for value, character in encode.items()}


_BYTE_ENCODE, _BYTE_DECODE = _byte_maps()


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_space(character: str) -> bool:
    return character.isspace()


def _pretokenize(text: str) -> Iterable[str]:
    """Implement Qwen3's published split regex without a regex dependency."""

    index = 0
    length = len(text)
    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    while index < length:
        lowered = text[index : index + 3].lower()
        contraction = next(
            (item for item in contractions if lowered.startswith(item)), None
        )
        if contraction is not None:
            end = index + len(contraction)
            yield text[index:end]
            index = end
            continue

        current = text[index]
        if _is_letter(current):
            end = index + 1
            while end < length and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue
        if (
            current not in "\r\n"
            and not _is_letter(current)
            and not _is_number(current)
            and index + 1 < length
            and _is_letter(text[index + 1])
        ):
            end = index + 2
            while end < length and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue

        if _is_number(current):
            yield current
            index += 1
            continue

        punctuation_start = index
        if current == " " and index + 1 < length:
            following = text[index + 1]
            if not _is_space(following) and not _is_letter(following) and not _is_number(following):
                index += 1
                current = following
        if not _is_space(current) and not _is_letter(current) and not _is_number(current):
            index += 1
            while index < length:
                candidate = text[index]
                if _is_space(candidate) or _is_letter(candidate) or _is_number(candidate):
                    break
                index += 1
            while index < length and text[index] in "\r\n":
                index += 1
            yield text[punctuation_start:index]
            continue
        index = punctuation_start

        if _is_space(text[index]):
            end = index
            last_newline = -1
            while end < length and _is_space(text[end]):
                if text[end] in "\r\n":
                    last_newline = end
                end += 1
            if last_newline >= index:
                end = last_newline + 1
            yield text[index:end]
            index = end
            continue

        raise Qwen3TokenizerError(f"pretokenizer made no progress at character {index}")


class Qwen3NativeTokenizer:
    """Hash-bound Qwen3 tokenizer with no runtime package dependencies."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        tokenizer_path = self.model_path / "tokenizer.json"
        config_path = self.model_path / "tokenizer_config.json"
        document, tokenizer_sha256, tokenizer_bytes = _read_json(tokenizer_path)
        config, config_sha256, config_bytes = _read_json(config_path)
        self._validate_pipeline(document, config)

        model = document["model"]
        self.vocab = {str(token): int(token_id) for token, token_id in model["vocab"].items()}
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        if len(self.id_to_token) != len(self.vocab):
            raise Qwen3TokenizerError("tokenizer vocabulary contains duplicate IDs")
        self.merge_ranks = {
            (str(pair[0]), str(pair[1])): rank
            for rank, pair in enumerate(model["merges"])
        }
        if len(self.merge_ranks) != len(model["merges"]):
            raise Qwen3TokenizerError("tokenizer contains duplicate BPE merges")
        self.added_to_id = {
            str(item["content"]): int(item["id"])
            for item in document["added_tokens"]
        }
        self.special_to_id = {
            str(item["content"]): int(item["id"])
            for item in document["added_tokens"]
            if item.get("special") is True
        }
        self.special_ids = frozenset(self.special_to_id.values())
        self._added_by_id = {token_id: text for text, token_id in self.added_to_id.items()}
        if len(self._added_by_id) != len(self.added_to_id):
            raise Qwen3TokenizerError("tokenizer added tokens contain duplicate IDs")
        self._added_tokens = tuple(sorted(self.added_to_id, key=len, reverse=True))
        implementation = Path(__file__).read_bytes()
        self._binding = {
            "schema": "hive.qwen3-native-tokenizer.v1",
            "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
            "tokenizer_json_sha256": tokenizer_sha256,
            "tokenizer_json_bytes": tokenizer_bytes,
            "tokenizer_config_sha256": config_sha256,
            "tokenizer_config_bytes": config_bytes,
            "vocab_entries": len(self.vocab),
            "merge_entries": len(self.merge_ranks),
            "added_tokens": len(self.added_to_id),
            "special_tokens": len(self.special_to_id),
            "normalization": "NFC",
            "pretokenizer": "qwen3-published-split-plus-bytelevel",
            "chat_contract": "single-user-add-generation-prompt-no-thinking-v1",
        }

    @staticmethod
    def _validate_pipeline(document: dict[str, Any], config: dict[str, Any]) -> None:
        model = document.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise Qwen3TokenizerError("tokenizer model must be BPE")
        if model.get("dropout") is not None or model.get("byte_fallback") is not False:
            raise Qwen3TokenizerError("unsupported stochastic or byte-fallback tokenizer model")
        if not isinstance(model.get("vocab"), dict) or not isinstance(model.get("merges"), list):
            raise Qwen3TokenizerError("tokenizer vocabulary or merges are missing")
        normalizer = document.get("normalizer")
        if normalizer != {"type": "NFC"}:
            raise Qwen3TokenizerError("tokenizer normalizer must be exactly NFC")
        pre = document.get("pre_tokenizer")
        if not isinstance(pre, dict) or pre.get("type") != "Sequence":
            raise Qwen3TokenizerError("tokenizer pre-tokenizer must be a sequence")
        parts = pre.get("pretokenizers")
        if not isinstance(parts, list) or len(parts) != 2:
            raise Qwen3TokenizerError("tokenizer pre-tokenizer sequence is unsupported")
        split, bytelevel = parts
        pattern = split.get("pattern") if isinstance(split, dict) else None
        if (
            split.get("type") != "Split"
            or split.get("behavior") != "Isolated"
            or not isinstance(pattern, dict)
            or pattern.get("Regex")
            != r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ):
            raise Qwen3TokenizerError("tokenizer split pattern is not the Qwen3 contract")
        expected_bytelevel = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        if bytelevel != expected_bytelevel or document.get("decoder") != expected_bytelevel:
            raise Qwen3TokenizerError("tokenizer ByteLevel contract changed")
        added = document.get("added_tokens")
        if not isinstance(added, list):
            raise Qwen3TokenizerError("tokenizer added tokens are missing")
        specials = {item.get("content"): item.get("id") for item in added if item.get("special") is True}
        for token, token_id in {
            "<|endoftext|>": 151643,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        }.items():
            if specials.get(token) != token_id:
                raise Qwen3TokenizerError(f"required Qwen3 special token changed: {token}")
        template = config.get("chat_template")
        if not isinstance(template, str) or any(fragment not in template for fragment in _REQUIRED_TEMPLATE_FRAGMENTS):
            raise Qwen3TokenizerError("Qwen3 chat template no longer supports the locked shape")
        if config.get("tokenizer_class") != "Qwen2Tokenizer":
            raise Qwen3TokenizerError("unexpected Qwen tokenizer class declaration")

    def binding(self) -> dict[str, Any]:
        return dict(self._binding)

    @lru_cache(maxsize=65536)
    def _bpe(self, token: str) -> tuple[str, ...]:
        symbols = list(token)
        if len(symbols) < 2:
            return tuple(symbols)
        while True:
            candidate_rank: int | None = None
            candidate_pair: tuple[str, str] | None = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (candidate_rank is None or rank < candidate_rank):
                    candidate_rank = rank
                    candidate_pair = pair
            if candidate_pair is None:
                return tuple(symbols)
            merged = candidate_pair[0] + candidate_pair[1]
            result: list[str] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == candidate_pair:
                    result.append(merged)
                    index += 2
                else:
                    result.append(symbols[index])
                    index += 1
            symbols = result

    def _ordinary_ids(self, text: str) -> list[int]:
        result: list[int] = []
        for piece in _pretokenize(text):
            byte_token = "".join(_BYTE_ENCODE[value] for value in piece.encode("utf-8"))
            for merged in self._bpe(byte_token):
                token_id = self.vocab.get(merged)
                if token_id is None:
                    raise Qwen3TokenizerError(f"BPE produced an unknown vocabulary token: {merged!r}")
                result.append(token_id)
        return result

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Qwen3 tokenizer input must be a string")
        normalized = unicodedata.normalize("NFC", text)
        result: list[int] = []
        ordinary_start = 0
        index = 0
        while index < len(normalized):
            added = next(
                (token for token in self._added_tokens if normalized.startswith(token, index)),
                None,
            )
            if added is None:
                index += 1
                continue
            if ordinary_start < index:
                result.extend(self._ordinary_ids(normalized[ordinary_start:index]))
            result.append(self.added_to_id[added])
            index += len(added)
            ordinary_start = index
        if ordinary_start < len(normalized):
            result.extend(self._ordinary_ids(normalized[ordinary_start:]))
        return result

    def prompt_ids(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        return self.encode(_CHAT_PREFIX + prompt + _CHAT_MIDDLE + _NO_THINK)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        encoded = bytearray()
        for raw_id in token_ids:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise Qwen3TokenizerError("token IDs must be integers")
            if raw_id in self.special_ids and skip_special_tokens:
                continue
            added = self._added_by_id.get(raw_id)
            if added is not None:
                encoded.extend(added.encode("utf-8"))
                continue
            token = self.id_to_token.get(raw_id)
            if token is None:
                raise Qwen3TokenizerError(f"unknown token ID: {raw_id}")
            try:
                encoded.extend(_BYTE_DECODE[character] for character in token)
            except KeyError as exc:
                raise Qwen3TokenizerError(f"token contains a non-ByteLevel character: {token!r}") from exc
        return encoded.decode("utf-8", errors="replace")


__all__ = ["Qwen3NativeTokenizer", "Qwen3TokenizerError"]
