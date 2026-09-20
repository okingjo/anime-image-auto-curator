"""Metadata / prompt extraction for SD-webui & ComfyUI images (PNG + JPEG).

Verified against real samples:
- JPEG: prompt is in EXIF UserComment, encoded UTF-16-BE after an 8-byte charset code.
- PNG (SD-webui): prompt is in the 'parameters' tEXt chunk (same block format).
- PNG (ComfyUI native): 'prompt' / 'workflow' JSON chunks (best-effort).

Block format (webui 'parameters'):
    <positive prompt>
    Negative prompt: <negative>
    Steps: N, Sampler: ..., CFG scale: ..., Seed: ..., Size: WxH, Model: ..., Version: ...
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# EXIF UserComment charset codes occupy the first 8 bytes.
_UC_CODES = (b"ASCII\x00\x00\x00", b"UNICODE\x00", b"JIS\x00\x00\x00\x00\x00")


@dataclass
class ImageMeta:
    path: str
    filename: str
    width: int = 0
    height: int = 0
    positive: str = ""
    negative: str = ""
    params: dict = field(default_factory=dict)
    loras: list = field(default_factory=list)  # list[(name, weight)]
    source: str = "none"  # exif-usercomment | png-parameters | comfyui-json | none
    raw: str = ""

    @property
    def seed(self):
        return self.params.get("Seed") or self.params.get("seed")

    @property
    def model(self):
        return self.params.get("Model") or self.params.get("model")


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _decode_usercomment(raw: bytes) -> str:
    """Decode an EXIF UserComment, handling ASCII / UNICODE(UTF-16) / JIS."""
    if not raw:
        return ""
    head = raw[:8]
    body = raw[8:] if any(head.startswith(c[: len(c)]) for c in _UC_CODES) else raw

    if head.startswith(b"ASCII"):
        return body.decode("latin-1", "replace")
    if head.startswith(b"JIS"):
        try:
            return body.decode("shift_jis", "replace")
        except Exception:
            return body.decode("latin-1", "replace")
    # UNICODE (or unknown): try BOM, then heuristic UTF-16 BE vs LE, then UTF-8/latin.
    if body[:2] == b"\xff\xfe":
        return body[2:].decode("utf-16-le", "replace")
    if body[:2] == b"\xfe\xff":
        return body[2:].decode("utf-16-be", "replace")

    def _printable(s: str) -> int:
        return sum(1 for c in s if 32 <= ord(c) < 127)

    try:
        be = body.decode("utf-16-be", "replace")
        le = body.decode("utf-16-le", "replace")
        best = be if _printable(be) >= _printable(le) else le
        if _printable(best) >= max(8, len(best) // 4):
            return best
    except Exception:
        pass
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", "replace")


# --------------------------------------------------------------------------- #
# Block parsing
# --------------------------------------------------------------------------- #
_LORA_RE = re.compile(r"<lora:([^:>]+):([^>]+)>", re.IGNORECASE)


def parse_loras(text: str) -> list:
    unesc = text.replace("\\<", "<").replace("\\>", ">")
    out, seen = [], set()
    for name, weight in _LORA_RE.findall(unesc):
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        try:
            w = float(weight)
        except ValueError:
            w = None
        out.append((name, w))
    return out


def _parse_params_line(line: str) -> dict:
    """Parse 'Steps: 40, Sampler: Euler a, CFG scale: 5.5, Seed: 123, ...'."""
    params = {}
    # Split on commas that precede a 'Key:' token.
    parts = re.split(r",\s*(?=[A-Za-z][A-Za-z _/]*:)", line)
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        params[key.strip()] = val.strip()
    return params


def parse_parameters_block(text: str):
    """Split a webui 'parameters' block into (positive, negative, params_dict)."""
    text = text.strip()
    positive, negative, params = text, "", {}

    neg_split = re.split(r"\n?Negative prompt:", text, maxsplit=1)
    if len(neg_split) == 2:
        positive = neg_split[0]
        rest = neg_split[1]
        step_split = re.split(r"\n?Steps:", rest, maxsplit=1)
        negative = step_split[0].strip()
        if len(step_split) == 2:
            params = _parse_params_line("Steps:" + step_split[1])
    else:
        step_split = re.split(r"\n?Steps:", text, maxsplit=1)
        positive = step_split[0]
        if len(step_split) == 2:
            params = _parse_params_line("Steps:" + step_split[1])

    return positive.strip(), negative.strip(), params


# --------------------------------------------------------------------------- #
# Tag tokenisation (for outfit / fidelity comparison)
# --------------------------------------------------------------------------- #
def split_tags(prompt: str) -> list:
    """Split a prompt into clean, lowercased tag tokens (paren/weight aware)."""
    tokens, depth, cur = [], 0, []
    for ch in prompt:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            tokens.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))

    out = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # unwrap one layer of emphasis parens, drop trailing :weight
        t = re.sub(r"^\((.*)\)$", r"\1", t)
        t = re.sub(r":\s*-?[\d.]+\s*$", "", t)
        t = t.replace("\\(", "(").replace("\\)", ")").replace("\\<", "<").replace("\\>", ">")
        t = t.strip().lower()
        if t:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# ComfyUI JSON (best-effort)
# --------------------------------------------------------------------------- #
def _comfy_texts(prompt_json: str):
    """Extract CLIPTextEncode 'text' fields from a ComfyUI API prompt JSON."""
    try:
        graph = json.loads(prompt_json)
    except Exception:
        return []
    nodes = graph if isinstance(graph, dict) else {}
    # API format: {node_id: {class_type, inputs:{text:...}}}
    texts = []
    for node in nodes.values():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            txt = (node.get("inputs") or {}).get("text")
            if isinstance(txt, str) and txt.strip():
                texts.append(txt.strip())
    return texts


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
def read_meta(path: str | Path) -> ImageMeta:
    path = Path(path)
    meta = ImageMeta(path=str(path), filename=path.name)

    try:
        with Image.open(path) as img:
            meta.width, meta.height = img.size
            info = dict(img.info)
            exif = img.getexif()
    except Exception:
        return meta

    block = None

    # 1) SD-webui PNG 'parameters'
    if isinstance(info.get("parameters"), str):
        block = info["parameters"]
        meta.source = "png-parameters"

    # 2) JPEG (or PNG) EXIF UserComment
    if block is None and exif:
        try:
            sub = exif.get_ifd(0x8769)  # Exif sub-IFD
            uc = sub.get(0x9286)  # UserComment
            if uc is not None:
                raw = bytes(uc) if not isinstance(uc, str) else uc.encode("latin-1", "replace")
                decoded = _decode_usercomment(raw)
                if decoded.strip():
                    block = decoded
                    meta.source = "exif-usercomment"
        except Exception:
            pass

    # 3) ComfyUI native PNG JSON
    if block is None and isinstance(info.get("prompt"), str):
        texts = _comfy_texts(info["prompt"])
        if texts:
            # Heuristic: join all CLIPTextEncode texts; first ~ positive.
            block = "\n".join(texts)
            meta.source = "comfyui-json"

    if block:
        meta.raw = block
        meta.positive, meta.negative, meta.params = parse_parameters_block(block)
        meta.loras = parse_loras(block)

    return meta


# --------------------------------------------------------------------------- #
# Grouping key
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = s.lower().replace("\\", "")
    s = re.sub(r"<lora:[^>]*>", "", s)  # LoRA weight/name shouldn't split a batch
    s = re.sub(r"\s+", " ", s).strip()
    return s


def group_key(meta: ImageMeta) -> str:
    """Stable key so images from the same prompt batch (diff seeds) cluster."""
    key = _norm(meta.positive)
    if meta.negative:
        key += "||" + _norm(meta.negative)
    return key or "(no prompt)"


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT
