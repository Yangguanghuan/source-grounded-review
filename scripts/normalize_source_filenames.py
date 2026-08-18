from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from source_grounded_review.core.utils import safe_name, write_csv
from source_grounded_review.io.corpus import attach_document_paths, iter_document_paths, load_references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy matched source files to normalized ref_id_title filenames.")
    parser.add_argument("--references", required=True, help="Reference CSV.")
    parser.add_argument("--input-dir", "--paper-dir", dest="input_dir", required=True, help="Input source directory.")
    parser.add_argument("--out-dir", required=True, help="Output normalized source directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    references_path = Path(args.references)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    references = attach_document_paths(load_references(references_path), input_dir)
    copied_sources: set[Path] = set()
    manifest_rows: list[dict[str, str]] = []

    for ref in references:
        if not ref.document_path:
            manifest_rows.append(ref_row(ref, status="unmatched_reference"))
            continue
        source = Path(ref.document_path)
        if not source.exists():
            manifest_rows.append(ref_row(ref, status="missing_source"))
            continue
        target = unique_target_path(out_dir, ref.ref_id, ref.title, source.suffix)
        shutil.copy2(source, target)
        copied_sources.add(source.resolve())
        manifest_rows.append(
            {
                **ref_row(ref, status="copied"),
                "source_path": str(source),
                "normalized_path": str(target),
                "normalized_name": target.name,
            }
        )

    for source in iter_document_paths(input_dir):
        if source.resolve() not in copied_sources:
            manifest_rows.append(
                {
                    "status": "unmatched_file",
                    "ref_id": "",
                    "title": "",
                    "source_path": str(source),
                    "normalized_path": "",
                    "normalized_name": "",
                }
            )

    manifest_path = write_csv(out_dir / "normalization_manifest.csv", manifest_rows)
    print(f"normalized_dir={out_dir}")
    print(f"manifest={manifest_path}")
    print(f"copied={sum(1 for row in manifest_rows if row.get('status') == 'copied')}")
    print(f"unmatched_references={sum(1 for row in manifest_rows if row.get('status') == 'unmatched_reference')}")
    print(f"unmatched_files={sum(1 for row in manifest_rows if row.get('status') == 'unmatched_file')}")


def ref_row(ref, *, status: str) -> dict[str, str]:
    return {
        "status": status,
        "ref_id": ref.ref_id,
        "title": ref.title,
        "doi": ref.doi,
        "source_path": ref.document_path,
        "normalized_path": "",
        "normalized_name": "",
    }


def unique_target_path(out_dir: Path, ref_id: str, title: str, suffix: str) -> Path:
    ref_prefix = ref_id.zfill(3) if ref_id.isdigit() else safe_name(ref_id, 20)
    base = safe_name(f"{ref_prefix}_{title}", 170)
    target = out_dir / f"{base}{suffix.lower()}"
    if not target.exists():
        return target
    for index in range(2, 1000):
        candidate = out_dir / f"{base}_{index}{suffix.lower()}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique filename for {base}")


if __name__ == "__main__":
    main()

