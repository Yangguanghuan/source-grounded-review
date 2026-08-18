from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from source_grounded_review.core.llm import LLMClient, MockLLM
from source_grounded_review.core.models import ClaimAudit, Evidence
from source_grounded_review.core.utils import clean_text, split_sentences, token_score


CITATION_RE = re.compile(r"\[([A-Za-z]?\d+(?:\s*[,，、]\s*[A-Za-z]?\d+)*)\]")


def audit_draft(draft_markdown: str, evidence_rows: list[Evidence], llm: LLMClient | None = None) -> list[ClaimAudit]:
    evidence_by_ref: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence_rows:
        evidence_by_ref[item.ref_id].append(item)

    audits: list[ClaimAudit] = []
    section = ""
    for line in draft_markdown.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
            continue
        if section in {"证据链说明", "Evidence Traceability Notes"}:
            continue
        for sentence in split_sentences(stripped):
            if should_skip_sentence(sentence):
                continue
            citations = extract_citations(sentence)
            if not citations:
                audits.append(
                    ClaimAudit(
                        claim_id=f"C{len(audits) + 1:05d}",
                        section=section,
                        claim_text=sentence,
                        citation_ids=[],
                        verdict="needs_more_citation",
                        best_evidence_ids=[],
                        relevance_score=0.0,
                        rationale="该 claim 没有检测到引用。",
                    )
                )
                continue
            candidates = [item for ref_id in citations for item in evidence_by_ref.get(ref_id, [])]
            best = sorted(candidates, key=lambda item: token_score(strip_citations(sentence), item.evidence_text), reverse=True)
            best_score = token_score(strip_citations(sentence), best[0].evidence_text) if best else 0.0
            if llm is not None and not isinstance(llm, MockLLM) and best:
                verdict, rationale = model_verdict(sentence, citations, best[:5], best_score, llm)
            else:
                verdict = classify(best_score, bool(best))
                rationale = rationale_for(verdict, best_score)
            audits.append(
                ClaimAudit(
                    claim_id=f"C{len(audits) + 1:05d}",
                    section=section,
                    claim_text=clean_text(sentence),
                    citation_ids=citations,
                    verdict=verdict,
                    best_evidence_ids=[item.evidence_id for item in best[:3]],
                    relevance_score=best_score,
                    rationale=rationale,
                )
            )
    return audits


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for match in CITATION_RE.finditer(text):
        raw = match.group(1).replace("，", ",").replace("、", ",")
        for part in raw.split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return sorted(set(ids), key=str)


def should_skip_sentence(sentence: str) -> bool:
    text = sentence.strip()
    return any(
        marker in text
        for marker in [
            "本节当前缺少可用证据",
            "本节当前没有分配到证据包",
            "本节只能基于上述证据进行有限概括",
        ]
    )


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def classify(score: float, has_evidence: bool) -> str:
    if not has_evidence:
        return "unknown_citation"
    if score >= 0.18:
        return "supports"
    if score >= 0.08:
        return "partially_supports"
    return "misaligned"


def rationale_for(verdict: str, score: float) -> str:
    if verdict == "supports":
        return f"被引用文献证据与 claim 有较高文本重叠，score={score:.3f}，仍需人工核对语义强度。"
    if verdict == "partially_supports":
        return f"存在相关证据，但支持强度中等，score={score:.3f}。"
    if verdict == "misaligned":
        return f"被引用文献证据与 claim 相关性较弱，score={score:.3f}。"
    return "未找到该引用编号对应的证据。"


def model_verdict(
    claim: str,
    citations: list[str],
    evidence: list[Evidence],
    heuristic_score: float,
    llm: LLMClient,
) -> tuple[str, str]:
    system = (
        "You are a conservative citation evidence auditor. Return strict JSON only. "
        "Use only the provided evidence snippets."
    )
    user = (
        "请判断综述 claim 是否被其引用文献证据支持。只能依据 evidence，不得补充外部知识。\n"
        "verdict 只能是 supports, partially_supports, misaligned, overstated, needs_more_citation。\n"
        "返回 JSON: {\"verdict\":\"...\", \"rationale\":\"...\"}\n\n"
        f"claim: {claim}\n"
        f"citations: {citations}\n"
        "evidence:\n"
        + "\n".join(
            f"- evidence_id={item.evidence_id}; ref_id={item.ref_id}; page={item.page_hint}; text={item.evidence_text}"
            for item in evidence
        )
    )
    try:
        data = parse_json_object(llm.complete(system, user, temperature=0.0))
    except Exception as exc:  # noqa: BLE001 - keep pipeline usable if audit call fails.
        verdict = classify(heuristic_score, bool(evidence))
        return verdict, f"模型审查失败，已回退本地判定：{exc}"
    verdict = str(data.get("verdict", "")).strip()
    if verdict not in {"supports", "partially_supports", "misaligned", "overstated", "needs_more_citation"}:
        verdict = classify(heuristic_score, bool(evidence))
    rationale = str(data.get("rationale", "")).strip() or rationale_for(verdict, heuristic_score)
    return verdict, rationale


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

