from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from source_grounded_review.core.llm import LLMClient, MockLLM
from source_grounded_review.core.models import Evidence, OutlineSection, SourceCard, SourceSectionAssignment
from source_grounded_review.core.utils import clean_text, token_score


@dataclass
class SectionSelectionConfig:
    max_refs_per_section: int = 6
    max_evidence_per_section: int = 10
    max_evidence_per_ref_section: int = 2
    use_model: bool = False
    model_candidate_limit: int = 20


def select_section_evidence(
    sections: list[OutlineSection],
    assignments: list[SourceSectionAssignment],
    evidence_rows: list[Evidence],
    cards: list[SourceCard],
    config: SectionSelectionConfig | None = None,
    llm: LLMClient | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    config = config or SectionSelectionConfig()
    evidence_by_id = {item.evidence_id: item for item in evidence_rows}
    cards_by_ref = {card.ref_id: card for card in cards}
    claims_by_evidence_id = build_claim_lookup(cards)

    assignments_by_section: dict[str, list[SourceSectionAssignment]] = {}
    for assignment in assignments:
        assignments_by_section.setdefault(assignment.section_id, []).append(assignment)

    selected_by_section: dict[str, list[str]] = {}
    ranking_rows: list[dict[str, object]] = []
    for section in sections:
        candidates = collect_candidates(section.section_id, assignments_by_section, evidence_by_id)
        scored = sorted(
            (
                score_candidate(section, assignment, evidence, cards_by_ref.get(evidence.ref_id), claims_by_evidence_id)
                for assignment, evidence in candidates
            ),
            key=lambda row: row["selection_score"],
            reverse=True,
        )
        selection_mode = "rule"
        model_rationales: dict[str, str] = {}
        if config.use_model and llm is not None and not isinstance(llm, MockLLM) and scored:
            selected_ids, model_rationales = choose_by_model(section, scored, config, llm)
            if selected_ids:
                selection_mode = "model"
            else:
                selected_ids = choose_diverse(scored, config)
        else:
            selected_ids = choose_diverse(scored, config)
        selected_by_section[section.section_id] = selected_ids
        for rank, row in enumerate(scored, start=1):
            evidence = row["evidence"]
            claim = row["claim"]
            ranking_rows.append(
                {
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "candidate_rank": rank,
                    "selected": "yes" if evidence.evidence_id in selected_ids else "no",
                    "selection_mode": selection_mode,
                    "selection_score": round(float(row["selection_score"]), 4),
                    "ref_id": evidence.ref_id,
                    "evidence_id": evidence.evidence_id,
                    "title": evidence.title,
                    "page_hint": evidence.page_hint,
                    "role": row["assignment"].role,
                    "support_strength": row["assignment"].support_strength,
                    "evidence_type": claim.get("evidence_type", ""),
                    "conclusion_strength": claim.get("conclusion_strength", ""),
                    "reason": row["reason"],
                    "model_rationale": model_rationales.get(evidence.evidence_id, ""),
                    "claim": claim.get("claim", ""),
                    "evidence_text": evidence.evidence_text,
                }
            )
    return selected_by_section, ranking_rows


def build_claim_lookup(cards: list[SourceCard]) -> dict[str, dict[str, object]]:
    claims: dict[str, dict[str, object]] = {}
    for card in cards:
        for claim in card.evidence_claims:
            evidence_id = str(claim.get("evidence_id", "")).strip()
            if evidence_id:
                claims[evidence_id] = claim
    return claims


def collect_candidates(
    section_id: str,
    assignments_by_section: dict[str, list[SourceSectionAssignment]],
    evidence_by_id: dict[str, Evidence],
) -> list[tuple[SourceSectionAssignment, Evidence]]:
    candidates: list[tuple[SourceSectionAssignment, Evidence]] = []
    seen: set[str] = set()
    for assignment in assignments_by_section.get(section_id, []):
        for evidence_id in assignment.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence_id in seen:
                continue
            seen.add(evidence_id)
            candidates.append((assignment, evidence))
    return candidates


def score_candidate(
    section: OutlineSection,
    assignment: SourceSectionAssignment,
    evidence: Evidence,
    card: SourceCard | None,
    claims_by_evidence_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    claim = claims_by_evidence_id.get(evidence.evidence_id, {})
    evidence_type = str(claim.get("evidence_type", ""))
    conclusion_strength = str(claim.get("conclusion_strength", ""))
    study_type = (card.study_type if card else "").lower()

    section_match = token_score(section.title, evidence.evidence_text)
    score = evidence.score * 0.25 + section_match * 0.55
    score += {"strong": 0.25, "moderate": 0.14, "context": 0.04}.get(assignment.support_strength, 0.0)
    score += {"direct": 0.25, "indirect": 0.12, "context": 0.04}.get(conclusion_strength, 0.0)
    type_bonus, type_reason = section_type_bonus(section.title, evidence_type, assignment.role, study_type)
    score += type_bonus
    if evidence.page_hint:
        score += 0.03

    reason = (
        f"section_match={section_match:.3f}; evidence_score={evidence.score:.3f}; "
        f"assignment={assignment.support_strength}; conclusion={conclusion_strength or 'unknown'}; "
        f"{type_reason}"
    )
    return {
        "assignment": assignment,
        "evidence": evidence,
        "claim": claim,
        "selection_score": score,
        "reason": reason,
    }


def section_type_bonus(section_title: str, evidence_type: str, role: str, study_type: str) -> tuple[float, str]:
    title = section_title.lower()
    bonus = 0.0
    reasons: list[str] = []
    if any(term in section_title for term in ["背景", "意义", "研究现状", "文献综述", "理论框架"]):
        if evidence_type == "background":
            bonus += 0.18
            reasons.append("background_evidence_match")
        if "review" in study_type:
            bonus += 0.16
            reasons.append("background_prefers_review")
        if evidence_type in {"design", "mechanism", "application"}:
            bonus += 0.08
            reasons.append("background_accepts_core_evidence")
    if any(term in section_title for term in ["设计", "结构", "框架", "模型", "架构", "体系"]):
        if evidence_type in {"design", "implementation"} or role == "method_or_implementation":
            bonus += 0.22
            reasons.append("design_section_match")
    if any(term in section_title for term in ["流程", "过程", "路径", "转化", "执行"]):
        if evidence_type in {"process", "mechanism", "implementation"}:
            bonus += 0.22
            reasons.append("process_section_match")
    if any(term in section_title for term in ["机制", "原理", "逻辑", "影响因素"]):
        if evidence_type == "mechanism" or role == "mechanism":
            bonus += 0.24
            reasons.append("mechanism_section_match")
    if any(term in section_title for term in ["应用", "效果", "转化", "安全", "实证", "分析"]):
        if evidence_type in {"application", "safety", "limitation", "evaluation"} or role == "application":
            bonus += 0.2
            reasons.append("application_or_safety_match")
    if any(term in section_title for term in ["方法", "检索", "筛选", "评价维度", "技术路线", "实现", "流程", "架构"]):
        if evidence_type in {"method", "implementation"}:
            bonus += 0.2
            reasons.append("method_section_match")
    if any(term in section_title for term in ["风险", "治理", "合规", "审计", "安全"]):
        if evidence_type in {"governance", "safety", "limitation"}:
            bonus += 0.22
            reasons.append("governance_section_match")
    if any(term in section_title for term in ["价值", "业务", "场景", "应用"]):
        if evidence_type in {"application", "evaluation"}:
            bonus += 0.18
            reasons.append("business_value_match")
    if "review" not in study_type and any(term in title for term in ["机制", "应用", "效果", "分析"]):
        bonus += 0.06
        reasons.append("primary_evidence_preferred")
    return bonus, "+".join(reasons) or "general_relevance"


def choose_diverse(scored_rows: list[dict[str, object]], config: SectionSelectionConfig) -> list[str]:
    selected: list[str] = []
    refs: set[str] = set()
    per_ref: dict[str, int] = {}
    types: set[str] = set()

    for row in scored_rows:
        evidence = row["evidence"]
        claim = row["claim"]
        ref_id = evidence.ref_id
        evidence_type = str(claim.get("evidence_type", ""))
        if len(selected) >= config.max_evidence_per_section:
            break
        if ref_id not in refs and len(refs) >= config.max_refs_per_section:
            continue
        if per_ref.get(ref_id, 0) >= config.max_evidence_per_ref_section:
            continue
        selected.append(evidence.evidence_id)
        refs.add(ref_id)
        per_ref[ref_id] = per_ref.get(ref_id, 0) + 1
        if evidence_type:
            types.add(evidence_type)

    if len(selected) >= config.max_evidence_per_section:
        return selected

    for row in scored_rows:
        evidence = row["evidence"]
        claim = row["claim"]
        evidence_type = str(claim.get("evidence_type", ""))
        if len(selected) >= config.max_evidence_per_section:
            break
        if evidence.evidence_id in selected:
            continue
        if evidence_type and evidence_type in types:
            continue
        selected.append(evidence.evidence_id)
        if evidence_type:
            types.add(evidence_type)

    return selected


def choose_by_model(
    section: OutlineSection,
    scored_rows: list[dict[str, object]],
    config: SectionSelectionConfig,
    llm: LLMClient,
) -> tuple[list[str], dict[str, str]]:
    candidates = scored_rows[: config.model_candidate_limit]
    allowed_ids = {row["evidence"].evidence_id for row in candidates}
    system = (
        "You select evidence for one section of a source-grounded review. "
        "Select a small, representative, non-redundant evidence set for one outline section. "
        "Return strict JSON only."
    )
    user = (
        f"章节标题：{section.title}\n"
        f"最多选择 {config.max_refs_per_section} 份不同资料，"
        f"最多选择 {config.max_evidence_per_section} 条证据，"
        f"同一份资料在本节最多选择 {config.max_evidence_per_ref_section} 条证据。\n\n"
        "筛选原则：\n"
        "1. 优先选择与本节标题强相关、结论强度为 direct 的证据。\n"
        "2. 避免同一份资料重复表达同一观点。\n"
        "3. 背景/研究现状章节优先保留综述性或代表性证据；方法/机制/应用章节优先保留更直接的证据。\n"
        "4. 尽量覆盖背景、方法、设计、过程、应用、评价、风险和局限中的不同维度。\n"
        "5. 只能从候选 evidence_id 中选择，不得新增证据。\n\n"
        "返回 JSON 格式："
        "{\"selected_evidence_ids\":[\"E000001\"],"
        "\"rationale_by_evidence_id\":{\"E000001\":\"选择原因\"},"
        "\"section_strategy\":\"本节筛选策略\"}\n\n"
        "候选证据：\n"
        + "\n".join(format_candidate(row) for row in candidates)
    )
    try:
        data = parse_json_object(llm.complete(system, user, temperature=0.0))
    except Exception:
        return [], {}
    raw_ids = data.get("selected_evidence_ids", [])
    if not isinstance(raw_ids, list):
        return [], {}
    raw_rationales = data.get("rationale_by_evidence_id", {})
    rationales = {
        str(key): clean_text(str(value), 240)
        for key, value in raw_rationales.items()
    } if isinstance(raw_rationales, dict) else {}
    selected = enforce_selection_limits(
        [str(item).strip() for item in raw_ids if str(item).strip() in allowed_ids],
        candidates,
        config,
    )
    if not selected:
        return [], {}
    strategy = clean_text(str(data.get("section_strategy", "")), 240)
    for evidence_id in selected:
        if not rationales.get(evidence_id) and strategy:
            rationales[evidence_id] = strategy
    return selected, rationales


def format_candidate(row: dict[str, object]) -> str:
    evidence = row["evidence"]
    claim = row["claim"]
    return (
        f"- evidence_id={evidence.evidence_id}; ref_id={evidence.ref_id}; "
        f"score={float(row['selection_score']):.3f}; "
        f"role={row['assignment'].role}; support={row['assignment'].support_strength}; "
        f"type={claim.get('evidence_type', '')}; strength={claim.get('conclusion_strength', '')}; "
        f"page={evidence.page_hint}; claim={clean_text(str(claim.get('claim', '')), 220)}; "
        f"evidence={clean_text(evidence.evidence_text, 360)}"
    )


def enforce_selection_limits(
    evidence_ids: list[str],
    candidate_rows: list[dict[str, object]],
    config: SectionSelectionConfig,
) -> list[str]:
    evidence_by_id = {row["evidence"].evidence_id: row["evidence"] for row in candidate_rows}
    selected: list[str] = []
    refs: set[str] = set()
    per_ref: dict[str, int] = {}
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None or evidence_id in selected:
            continue
        if len(selected) >= config.max_evidence_per_section:
            break
        if evidence.ref_id not in refs and len(refs) >= config.max_refs_per_section:
            continue
        if per_ref.get(evidence.ref_id, 0) >= config.max_evidence_per_ref_section:
            continue
        selected.append(evidence_id)
        refs.add(evidence.ref_id)
        per_ref[evidence.ref_id] = per_ref.get(evidence.ref_id, 0) + 1
    return selected


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    inline = re.search(r"\{.*\}", text, flags=re.S)
    if inline:
        try:
            data = json.loads(inline.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

