from __future__ import annotations

from collections import defaultdict

from source_grounded_review.core.models import Evidence, OutlineSection, SourceCard, SourceSectionAssignment, SectionEvidencePack


def build_section_evidence_packs(
    sections: list[OutlineSection],
    assignments: list[SourceSectionAssignment],
    evidence_rows: list[Evidence],
    cards: list[SourceCard],
    selected_evidence_by_section: dict[str, list[str]] | None = None,
) -> list[SectionEvidencePack]:
    card_by_ref = {card.ref_id: card for card in cards}
    evidence_by_id = {item.evidence_id: item for item in evidence_rows}
    assignments_by_section: dict[tuple[str, str], list[SourceSectionAssignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_section[(assignment.section_id, assignment.section_title)].append(assignment)

    packs: list[SectionEvidencePack] = []
    for section in sections:
        section_id = section.section_id
        section_title = section.title
        section_assignments = assignments_by_section.get((section_id, section_title), [])
        if selected_evidence_by_section is not None:
            evidence_ids = [
                evidence_id
                for evidence_id in selected_evidence_by_section.get(section_id, [])
                if evidence_id in evidence_by_id
            ]
        else:
            evidence_ids = []
            for assignment in section_assignments:
                for evidence_id in assignment.evidence_ids:
                    if evidence_id and evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
        selected_refs = {evidence_by_id[evidence_id].ref_id for evidence_id in evidence_ids if evidence_id in evidence_by_id}

        source_summaries = []
        for assignment in section_assignments:
            if selected_evidence_by_section is not None and assignment.ref_id not in selected_refs:
                continue
            card = card_by_ref.get(assignment.ref_id)
            source_summaries.append(
                {
                    "ref_id": assignment.ref_id,
                    "title": assignment.title,
                    "role": assignment.role,
                    "support_strength": assignment.support_strength,
                    "reason": assignment.reason,
                    "topic_tags": "; ".join(card.topic_tags if card else []),
                    "method_tags": "; ".join(card.method_tags if card else []),
                    "use_case_tags": "; ".join(card.use_case_tags if card else []),
                    "key_findings": " | ".join(card.key_findings[:3] if card else []),
                }
            )

        packs.append(
            SectionEvidencePack(
                section_id=section_id,
                section_title=section_title,
                assigned_ref_ids=sorted(selected_refs or {assignment.ref_id for assignment in section_assignments}, key=str),
                evidence_ids=evidence_ids,
                source_summaries=source_summaries,
                evidence=[evidence_by_id[evidence_id].to_dict() for evidence_id in evidence_ids if evidence_id in evidence_by_id],
            )
        )
    return packs


def flatten_section_evidence_packs(packs: list[SectionEvidencePack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pack in packs:
        for evidence in pack.evidence:
            rows.append(
                {
                    "section_id": pack.section_id,
                    "section_title": pack.section_title,
                    "assigned_ref_ids": "; ".join(pack.assigned_ref_ids),
                    "evidence_id": evidence.get("evidence_id", ""),
                    "ref_id": evidence.get("ref_id", ""),
                    "title": evidence.get("title", ""),
                    "page_hint": evidence.get("page_hint", ""),
                    "score": evidence.get("score", ""),
                    "evidence_text": evidence.get("evidence_text", ""),
                }
            )
    return rows

