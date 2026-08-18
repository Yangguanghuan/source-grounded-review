from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]{1,}")
PAGE_RE = re.compile(r"\[PAGE\s+(\d+)\]")


def write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return str(path)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def tokenize(text: str) -> list[str]:
    tokens = [item.lower() for item in WORD_RE.findall(text)]
    for match in CJK_RE.findall(text):
        if len(match) == 1:
            tokens.append(match)
        else:
            tokens.extend(match[index : index + 2] for index in range(0, len(match) - 1))
    return [token for token in tokens if token not in STOPWORDS]


def token_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 220) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*", text)
    return [part.strip() for part in parts if len(part.strip()) >= 18]


def clean_text(text: str, limit: int | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if limit is not None:
        return cleaned[:limit]
    return cleaned


def page_hint(text: str) -> str:
    match = PAGE_RE.search(text)
    return match.group(1) if match else ""


def safe_name(text: str, limit: int = 120) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip(" ._") or "untitled"


def pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def flatten_list(items: Iterable[str]) -> str:
    return "; ".join(item for item in items if item)


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "can",
    "could",
    "研究",
    "显示",
    "表明",
    "通过",
    "以及",
    "具有",
    "可以",
    "可能",
    "不同",
    "进行",
}
