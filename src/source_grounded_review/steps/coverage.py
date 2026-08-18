from __future__ import annotations

import re
from collections import Counter

from source_grounded_review.core.models import (
    ClaimAudit,
    Document,
    Evidence,
    SourceCard,
    SourceSectionAssignment,
    SectionEvidencePack,
)


REF_RE = re.compile(r"\[([A-Za-z]?\d+(?:\s*[,，、]\s*[A-Za-z]?\d+)*)\]")


def build_reference_usage(
    documents: list[Document],
    cards: list[SourceCard],
    evidence_rows: list[Evidence],
    draft: str,
    audits: list[ClaimAudit],
    assignments: list[SourceSectionAssignment] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    evidence_count = Counter(item.ref_id for item in evidence_rows)
    cited_count = Counter(ref_id for ref_id in extract_all_citations(draft))
    weak_count = Counter(ref_id for audit in audits if audit.verdict != "supports" for ref_id in audit.citation_ids)
    assigned_count = Counter(item.ref_id for item in assignments or [])
    card_by_ref = {card.ref_id: card for card in cards}

    rows: list[dict[str, object]] = []
    for document in documents:
        ref = document.ref
        card = card_by_ref.get(ref.ref_id)
        rows.append(
            {
                "ref_id": ref.ref_id,
                "title": ref.title,
                "document_path": ref.document_path,
                "source_card": bool(card),
                "outline_assignment_count": assigned_count[ref.ref_id],
                "evidence_count": evidence_count[ref.ref_id],
                "draft_citation_count": cited_count[ref.ref_id],
                "weak_or_flagged_citation_count": weak_count[ref.ref_id],
                "suggested_sections": "; ".join(card.suggested_sections if card else []),
                "topic_tags": "; ".join(card.topic_tags if card else []),
                "method_tags": "; ".join(card.method_tags if card else []),
                "use_case_tags": "; ".join(card.use_case_tags if card else []),
                "status": "used" if cited_count[ref.ref_id] else "not_used_in_draft",
            }
        )
    unused = [row for row in rows if row["draft_citation_count"] == 0]
    return rows, unused


def build_coverage_critique(
    documents: list[Document],
    assignments: list[SourceSectionAssignment],
    packs: list[SectionEvidencePack],
    usage_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    assigned_refs = {item.ref_id for item in assignments}
    all_refs = {document.ref.ref_id for document in documents}
    unassigned_refs = sorted(all_refs - assigned_refs, key=str)
    unused_refs = [str(row["ref_id"]) for row in usage_rows if int(row.get("draft_citation_count", 0) or 0) == 0]
    weak_sections = [
        {
            "section_id": pack.section_id,
            "section_title": pack.section_title,
            "assigned_ref_count": len(pack.assigned_ref_ids),
            "evidence_count": len(pack.evidence_ids),
            "status": "weak" if len(pack.assigned_ref_ids) < 3 or len(pack.evidence_ids) < 3 else "ok",
        }
        for pack in packs
    ]
    rows = [
        {
            "metric": "sources_total",
            "value": len(documents),
            "note": "用户提供且可读取全文的文献数。",
        },
        {
            "metric": "assigned_sources",
            "value": len(assigned_refs),
            "note": "进入大纲章节匹配的资料数。",
        },
        {
            "metric": "unassigned_sources",
            "value": len(unassigned_refs),
            "note": "; ".join(unassigned_refs),
        },
        {
            "metric": "not_used_in_draft",
            "value": len(unused_refs),
            "note": "; ".join(unused_refs),
        },
        {
            "metric": "weak_sections",
            "value": sum(1 for item in weak_sections if item["status"] == "weak"),
            "note": "; ".join(item["section_title"] for item in weak_sections if item["status"] == "weak"),
        },
    ]
    report = render_coverage_report(rows, weak_sections)
    return rows + weak_sections, report


def render_coverage_report(rows: list[dict[str, object]], weak_sections: list[dict[str, object]]) -> str:
    metric = {str(row["metric"]): row for row in rows if "metric" in row}
    lines = [
        "# Coverage Report",
        "",
        f"- Documents read: {metric.get('sources_total', {}).get('value', 0)}",
        f"- Assigned references: {metric.get('assigned_sources', {}).get('value', 0)}",
        f"- Unassigned references: {metric.get('unassigned_sources', {}).get('value', 0)}",
        f"- References not cited in draft: {metric.get('not_used_in_draft', {}).get('value', 0)}",
        f"- Weak sections: {metric.get('weak_sections', {}).get('value', 0)}",
        "",
        "## Section Coverage",
        "",
    ]
    for item in weak_sections:
        lines.append(
            f"- {item['section_id']} {item['section_title']}: "
            f"{item['assigned_ref_count']} refs, {item['evidence_count']} evidence, status={item['status']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report checks whether the provided sources entered outline assignments and draft citations. "
            "Uncited sources are not automatically wrong, but they are worth checking when full-corpus participation matters.",
            "",
        ]
    )
    return "\n".join(lines)


def extract_all_citations(text: str) -> list[str]:
    refs: list[str] = []
    for match in REF_RE.finditer(text):
        raw = match.group(1).replace("，", ",").replace("、", ",")
        refs.extend(part.strip() for part in raw.split(",") if part.strip())
    return refs

