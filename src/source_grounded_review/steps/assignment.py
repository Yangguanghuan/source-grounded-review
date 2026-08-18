from __future__ import annotations

from collections import defaultdict

from source_grounded_review.core.models import Evidence, OutlineSection, SourceCard, SourceSectionAssignment
from source_grounded_review.core.utils import token_score


def assign_source_cards_to_outline(
    cards: list[SourceCard],
    sections: list[OutlineSection],
    evidence_rows: list[Evidence],
    *,
    max_sections_per_source: int = 3,
) -> list[SourceSectionAssignment]:
    """Connect every source card to its best outline sections.

    The user-provided corpus is treated as pre-selected, so each source is
    assigned to at least one section even when lexical similarity is weak.
    """

    evidence_by_ref: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence_rows:
        evidence_by_ref[item.ref_id].append(item)

    assignments: list[SourceSectionAssignment] = []
    for card in cards:
        ranked_sections = sorted(
            sections,
            key=lambda section: source_section_score(card, section, evidence_by_ref.get(card.ref_id, [])),
            reverse=True,
        )
        selected = ranked_sections[: max(1, max_sections_per_source)]
        for rank, section in enumerate(selected, start=1):
            related_evidence = rank_evidence_for_section(
                section,
                evidence_by_ref.get(card.ref_id, []),
                limit=4,
            )
            score = source_section_score(card, section, related_evidence)
            assignments.append(
                SourceSectionAssignment(
                    ref_id=card.ref_id,
                    title=card.title,
                    section_id=section.section_id,
                    section_title=section.title,
                    role=infer_role(card, section),
                    support_strength=infer_strength(score, rank),
                    evidence_ids=[item.evidence_id for item in related_evidence],
                    reason=build_reason(card, section, score, rank),
                )
            )
    return assignments


def source_section_score(card: SourceCard, section: OutlineSection, evidence_rows: list[Evidence]) -> float:
    card_text = " ".join(
        [
            card.title,
            card.study_type,
            " ".join(card.topic_tags),
            " ".join(card.method_tags),
            " ".join(card.use_case_tags),
            " ".join(card.key_findings),
            " ".join(card.suggested_sections),
        ]
    )
    evidence_text = " ".join(item.evidence_text for item in evidence_rows[:4])
    return token_score(section.title, card_text) * 0.7 + token_score(section.title, evidence_text) * 0.3


def rank_evidence_for_section(section: OutlineSection, evidence_rows: list[Evidence], limit: int) -> list[Evidence]:
    ranked = sorted(
        evidence_rows,
        key=lambda item: token_score(section.title, f"{item.section} {item.evidence_text}") + item.score * 0.1,
        reverse=True,
    )
    return ranked[:limit]


def infer_role(card: SourceCard, section: OutlineSection) -> str:
    title = section.title
    text = " ".join([title, " ".join(card.topic_tags), " ".join(card.method_tags), " ".join(card.use_case_tags)])
    if any(term in text for term in ["方法", "实现", "流程", "架构", "method", "implementation", "workflow", "architecture"]):
        return "method_or_implementation"
    if any(term in text for term in ["机制", "原理", "路径", "mechanism", "pathway"]):
        return "mechanism"
    if any(term in text for term in ["应用", "场景", "案例", "效果", "application", "case", "scenario"]):
        return "application"
    if any(term in text for term in ["风险", "治理", "合规", "审计", "安全", "risk", "governance", "compliance", "audit"]):
        return "risk_or_governance"
    if card.study_type == "review":
        return "background_or_synthesis"
    return "supporting_evidence"


def infer_strength(score: float, rank: int) -> str:
    if rank == 1 or score >= 0.16:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "context"


def build_reason(card: SourceCard, section: OutlineSection, score: float, rank: int) -> str:
    tags = [*card.topic_tags[:2], *card.method_tags[:2], *card.use_case_tags[:2]]
    tag_text = ", ".join(tag for tag in tags if tag and tag != "unspecified")
    if not tag_text:
        tag_text = card.study_type
    return (
        f"Rank {rank}; score={score:.3f}. Source tags ({tag_text}) align with "
        f"outline section '{section.title}'."
    )

