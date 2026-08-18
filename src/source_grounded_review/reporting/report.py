from __future__ import annotations

from collections import Counter

from source_grounded_review.core.models import ClaimAudit, Document, Evidence


def render_run_summary(
    *,
    topic: str,
    documents: list[Document],
    evidence_rows: list[Evidence],
    audits: list[ClaimAudit],
    output_paths: dict[str, str],
) -> str:
    verdict_counts = Counter(audit.verdict for audit in audits)
    lines = [
        f"# Run Summary",
        "",
        f"- Topic: {topic}",
        f"- Documents read: {len(documents)}",
        f"- Evidence rows: {len(evidence_rows)}",
        f"- Audited claims: {len(audits)}",
        "",
        "## Citation Check Verdicts",
        "",
    ]
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines.extend(["", "## Key Outputs", ""])
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The draft is constrained to the provided source files and local evidence table. "
            "The check table helps trace claims back to source snippets, but important passages should still be verified against the original files.",
            "",
        ]
    )
    return "\n".join(lines)

