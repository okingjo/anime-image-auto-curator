"""Character & outfit definitions + prompt matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .metadata import ImageMeta, split_tags


@dataclass
class Outfit:
    name: str
    tags: list = field(default_factory=list)
    default: bool = False


@dataclass
class Character:
    id: str
    name_tags: list = field(default_factory=list)
    lora_patterns: list = field(default_factory=list)
    outfits: list = field(default_factory=list)  # list[Outfit]

    def default_outfit(self):
        for o in self.outfits:
            if o.default:
                return o
        return self.outfits[0] if self.outfits else None


def load_characters(path: str | Path) -> dict:
    """Return {character_id: Character}. Missing/empty file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    chars = {}
    for c in data.get("characters", []) or []:
        cid = c.get("id")
        if not cid:
            continue
        outfits = [
            Outfit(
                name=o.get("name", f"outfit{i}"),
                tags=[str(t).strip().lower() for t in (o.get("tags") or [])],
                default=bool(o.get("default", False)),
            )
            for i, o in enumerate(c.get("outfits") or [])
        ]
        chars[cid] = Character(
            id=cid,
            name_tags=[str(t).strip().lower() for t in (c.get("name_tags") or [])],
            lora_patterns=[str(t).strip() for t in (c.get("lora_patterns") or [])],
            outfits=outfits,
        )
    return chars


def match_character(meta: ImageMeta, characters: dict) -> Character | None:
    pos = meta.positive.lower()
    lora_names = [name for name, _ in meta.loras]
    for ch in characters.values():
        if any(nt and nt in pos for nt in ch.name_tags):
            return ch
        if any(
            any(pat.lower() in ln.lower() for ln in lora_names)
            for pat in ch.lora_patterns
        ):
            return ch
    return None


def match_outfit(meta: ImageMeta, ch: Character) -> tuple:
    """Return (outfit_name_or_None, overlap_ratio, matched_tags, missing_tags).

    overlap_ratio = fraction of the outfit's tags present in the prompt.
    Best-matching outfit wins; if none reaches a minimal overlap, returns the
    default outfit with its (possibly low) ratio so the UI can flag anomalies.
    """
    if ch is None or not ch.outfits:
        return None, 0.0, [], []
    present = set(split_tags(meta.positive))
    best, best_ratio, best_match, best_miss = None, -1.0, [], []
    for o in ch.outfits:
        if not o.tags:
            continue
        oset = set(o.tags)
        matched = oset & present
        ratio = len(matched) / len(oset)
        if ratio > best_ratio:
            best, best_ratio = o, ratio
            best_match = sorted(matched)
            best_miss = sorted(oset - present)
    if best is None:
        best = ch.default_outfit()
        if best is None:
            return None, 0.0, [], []
        oset = set(best.tags)
        best_match = sorted(oset & present)
        best_miss = sorted(oset - present)
        best_ratio = (len(best_match) / len(oset)) if oset else 0.0
    return best.name, best_ratio, best_match, best_miss
