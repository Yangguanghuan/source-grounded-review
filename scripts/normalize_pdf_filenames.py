from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from source_grounded_review.core.utils import safe_name, write_csv
from source_grounded_review.io.corpus import iter_document_paths, preview_document_text


TITLE_SIGNAL_TERMS = {
    "review",
    "survey",
    "study",
    "analysis",
    "framework",
    "system",
    "method",
    "application",
    "evaluation",
    "研究",
    "综述",
    "分析",
    "框架",
    "方法",
    "应用",
    "评价",
}

HEADER_NOISE = {
    "abstract",
    "introduction",
    "review",
    "article",
    "research article",
    "open access",
    "citation",
    "published",
    "received",
    "accepted",
    "journal",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy every user PDF to Pxxx_title.pdf names inferred from PDF text.")
    parser.add_argument("--input-dir", "--paper-dir", dest="input_dir", required=True, help="Input PDF directory.")
    parser.add_argument("--out-dir", required=True, help="Output normalized PDF directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    paths = sorted(
        (path for path in iter_document_paths(input_dir) if path.suffix.lower() == ".pdf"),
        key=natural_sort_key,
    )
    for index, source in enumerate(paths, start=1):
        preview = preview_document_text(source, max_pages=2)
        title = infer_title(source, preview)
        ref_id = f"P{index:03d}"
        target = unique_target_path(out_dir, ref_id, title, source.suffix)
        shutil.copy2(source, target)
        rows.append(
            {
                "ref_id": ref_id,
                "inferred_title": title,
                "source_path": str(source),
                "normalized_path": str(target),
                "normalized_name": target.name,
            }
        )

    manifest_path = write_csv(out_dir / "normalization_manifest.csv", rows)
    print(f"normalized_dir={out_dir}")
    print(f"manifest={manifest_path}")
    print(f"copied={len(rows)}")


def infer_title(source: Path, preview: str) -> str:
    metadata_title = read_metadata_title(source)
    if looks_like_title(metadata_title):
        return clean_title(metadata_title)

    candidates = title_candidates(preview)
    if candidates:
        return clean_title(candidates[0])
    return source.stem.replace("_", " ").strip() or "untitled"


def read_metadata_title(source: Path) -> str:
    if source.suffix.lower() != ".pdf":
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        title = ""
        if reader.metadata:
            title = str(reader.metadata.get("/Title", "") or "")
        return title
    except Exception:
        return ""


def title_candidates(preview: str) -> list[str]:
    lines = [clean_title(line) for line in re.split(r"[\r\n]+", preview)]
    chunks = []
    for line in lines:
        if line:
            chunks.extend(split_long_line(line))
    scored = sorted(
        ((score_title_candidate(item), item) for item in chunks if looks_like_title(item)),
        reverse=True,
    )
    return [item for score, item in scored if score > 0]


def split_long_line(line: str) -> list[str]:
    if len(line) <= 180:
        return [line]
    parts = re.split(r"(?<=[.!?])\s+| {2,}", line)
    windows = []
    for part in parts:
        part = clean_title(part)
        if 20 <= len(part) <= 180:
            windows.append(part)
    if windows:
        return windows
    words = line.split()
    return [" ".join(words[start : start + 18]) for start in range(0, min(len(words), 60), 8)]


def looks_like_title(text: str) -> bool:
    cleaned = clean_title(text)
    lowered = cleaned.lower()
    if len(cleaned) < 18 or len(cleaned) > 190:
        return False
    if lowered in HEADER_NOISE:
        return False
    if lowered.startswith(("http", "doi:", "issn", "copyright", "received", "accepted", "published")):
        return False
    if sum(char.isalpha() for char in cleaned) < 12:
        return False
    return True


def score_title_candidate(text: str) -> float:
    lowered = text.lower()
    score = sum(term in lowered for term in TITLE_SIGNAL_TERMS) * 0.5
    score += 1.0 if 45 <= len(text) <= 150 else 0.0
    score += 0.5 if 4 <= len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)) <= 24 else 0.0
    score += 0.8 if ":" in text else 0.0
    score -= sum(term in lowered for term in HEADER_NOISE) * 0.7
    score -= 1.0 if "," in text and ";" in text else 0.0
    score -= 1.2 if re.search(r"\b\d{4}\b", text) else 0.0
    return score


def clean_title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = re.sub(r"^(research article|review|open access|article)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\|\s*.*$", "", cleaned)
    cleaned = cleaned.strip(" .,:;-_")
    return cleaned


def unique_target_path(out_dir: Path, ref_id: str, title: str, suffix: str) -> Path:
    base = safe_name(f"{ref_id}_{title}", 180)
    target = out_dir / f"{base}{suffix.lower()}"
    if not target.exists():
        return target
    for index in range(2, 1000):
        candidate = out_dir / f"{base}_{index}{suffix.lower()}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique filename for {base}")


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


if __name__ == "__main__":
    main()

