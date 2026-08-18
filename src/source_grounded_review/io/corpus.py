from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

from source_grounded_review.core.models import Document, Reference
from source_grounded_review.core.utils import pick, token_score


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
LEADING_NUMERIC_ID_RE = re.compile(r"^0*(\d+)(?:\D|$)")
SUPPORTED_TEXT_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
}


def load_corpus(references_csv: Path | None, input_dir: Path, max_sources: int | None = None) -> list[Document]:
    if references_csv:
        references = load_references(references_csv)
        references = attach_document_paths(references, input_dir)
    else:
        references = references_from_documents(input_dir)

    documents: list[Document] = []
    for ref in references:
        if not ref.document_path:
            continue
        path = Path(ref.document_path)
        if not path.exists():
            continue
        text = read_document_text(path)
        if not text.strip():
            continue
        documents.append(Document(ref=ref, text=text[:120000]))
        if max_sources is not None and len(documents) >= max_sources:
            break
    return documents


def load_references(path: Path) -> list[Reference]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    references: list[Reference] = []
    for index, row in enumerate(rows, start=1):
        ref_id = pick(row, "编号", "序号", "id", "ID", "ref_id", "Reference ID") or str(index)
        title = pick(row, "标题", "题名", "title", "Title")
        references.append(
            Reference(
                ref_id=str(ref_id).strip(),
                title=title,
                authors=pick(row, "作者", "authors", "Authors"),
                year=pick(row, "年份", "year", "Year"),
                source=pick(row, "来源", "期刊", "source", "journal", "Journal"),
                doi=pick(row, "DOI", "doi") or first_match(DOI_RE, " ".join(row.values())),
                keywords=pick(row, "关键词", "keywords", "Keywords"),
                link=pick(row, "来源链接", "链接", "url", "URL", "link"),
                document_path=pick(
                    row,
                    "文件路径",
                    "PDF路径",
                    "全文路径",
                    "资料路径",
                    "path",
                    "source_path",
                    "pdf_path",
                    "file_path",
                ),
                raw={str(key): str(value) for key, value in row.items()},
            )
        )
    return references


def references_from_documents(input_dir: Path) -> list[Reference]:
    references: list[Reference] = []
    for index, path in enumerate(iter_document_paths(input_dir), start=1):
        references.append(
            Reference(
                ref_id=f"P{index:03d}",
                title=path.stem.replace("_", " "),
                year=first_match(YEAR_RE, path.stem),
                document_path=str(path),
            )
        )
    return references


def attach_document_paths(references: list[Reference], input_dir: Path) -> list[Reference]:
    candidates = list(iter_document_paths(input_dir))
    previews = {path: preview_document_text(path) for path in candidates}
    used: set[Path] = set()
    for ref in references:
        explicit = Path(ref.document_path) if ref.document_path else None
        if explicit and explicit.exists():
            ref.document_path = str(explicit)
            used.add(explicit)
            continue
        exact_path = next(
            (path for path in candidates if path not in used and filename_matches_ref_id(ref.ref_id, path)),
            None,
        )
        if exact_path:
            ref.document_path = str(exact_path)
            used.add(exact_path)
            continue
        content_path = best_content_match(ref, candidates, previews, used)
        if content_path:
            ref.document_path = str(content_path)
            used.add(content_path)
            continue
        best_path: Path | None = None
        best_score = 0.0
        for path in candidates:
            if path in used:
                continue
            ref_id = (ref.ref_id or "").strip()
            if ref_id.isdigit() and leading_numeric_id(path) is not None:
                continue
            score = token_score(ref.title, path.stem)
            if ref.year and ref.year in path.stem:
                score += 0.08
            if score > best_score:
                best_score = score
                best_path = path
        if best_path and best_score >= 0.42:
            ref.document_path = str(best_path)
            used.add(best_path)
    return references


def best_content_match(
    ref: Reference,
    candidates: list[Path],
    previews: dict[Path, str],
    used: set[Path],
) -> Path | None:
    best_path: Path | None = None
    best_score = 0.0
    ref_doi = normalize_doi(ref.doi)
    for path in candidates:
        if path in used:
            continue
        preview = previews.get(path, "")
        if not preview:
            continue
        normalized_preview = preview.lower()
        score = token_score(ref.title, preview)
        if ref_doi and ref_doi in normalized_preview:
            score += 1.0
        if ref.year and ref.year in preview:
            score += 0.06
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_path and best_score >= 0.28 else None


def filename_matches_ref_id(ref_id: str, path: Path) -> bool:
    normalized = (ref_id or "").strip()
    if not normalized:
        return False
    stem = path.stem.strip()
    if normalized.isdigit():
        numeric_id = leading_numeric_id(path)
        return numeric_id == int(normalized)
    return bool(re.match(rf"^{re.escape(normalized)}(?:\W|_|$)", stem, re.IGNORECASE))


def leading_numeric_id(path: Path) -> int | None:
    match = LEADING_NUMERIC_ID_RE.match(path.stem.strip())
    return int(match.group(1)) if match else None


def iter_document_paths(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
            yield path


def preview_document_text(path: Path, max_pages: int = 2) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages[:max_pages]:
                pages.append(page.extract_text() or "")
            return clean_preview_text(" ".join(pages), 8000)
        except Exception:
            return ""
    try:
        return clean_preview_text(read_document_text(path), 8000)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF reading requires pypdf.") from exc
        reader = PdfReader(str(path))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(f"\n\n[PAGE {index}]\n{page.extract_text() or ''}")
        return "\n".join(pages)
    if suffix in {".json", ".jsonl"}:
        return read_json_text(path)
    if suffix in {".csv", ".tsv"}:
        return read_table_text(path)
    if suffix in {".html", ".htm"}:
        return read_html_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json_text(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        lines: list[str] = []
        for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                lines.append(f"[JSONL record {index}]\n{flatten_json_text(data)}")
            except json.JSONDecodeError:
                lines.append(stripped)
        return "\n\n".join(lines)
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return flatten_json_text(data)


def flatten_json_text(value: Any, prefix: str = "") -> str:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{label}:")
                lines.append(flatten_json_text(item, label))
            else:
                lines.append(f"{label}: {item}")
    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            label = f"{prefix}[{index}]" if prefix else f"item[{index}]"
            if isinstance(item, (dict, list)):
                lines.append(f"{label}:")
                lines.append(flatten_json_text(item, label))
            else:
                lines.append(f"{label}: {item}")
    else:
        lines.append(str(value))
    return "\n".join(line for line in lines if line.strip())


def read_table_text(path: Path) -> str:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames:
            rows = []
            for index, row in enumerate(reader, start=1):
                row_text = "; ".join(f"{key}={value}" for key, value in row.items() if value)
                if row_text:
                    rows.append(f"[ROW {index}] {row_text}")
            return "\n".join(rows)
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(re.sub(r"\s+", " ", raw)).strip()


def references_to_rows(documents: list[Document]) -> list[dict[str, str]]:
    rows = []
    for doc in documents:
        row = doc.ref.to_dict()
        row["text_chars"] = str(len(doc.text))
        rows.append(row)
    return rows


def first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(0) if match else ""


def clean_preview_text(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def normalize_doi(value: str) -> str:
    return (value or "").strip().lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")

