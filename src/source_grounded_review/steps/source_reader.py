from __future__ import annotations

import json
import re
from typing import Any

from source_grounded_review.core.llm import LLMClient, MockLLM
from source_grounded_review.core.models import Document, OutlineSection, SourceCard
from source_grounded_review.core.utils import chunk_text, clean_text, page_hint, split_sentences, token_score


TOPIC_TAG_TERMS = {
    "document management": ["document", "record", "ledger", "档案", "台账", "资料", "文档"],
    "retrieval": ["retrieval", "search", "semantic search", "检索", "搜索", "召回"],
    "workflow": ["workflow", "process", "automation", "流程", "过程", "自动化"],
    "governance": ["risk", "governance", "compliance", "audit", "风险", "治理", "合规", "审计"],
    "evaluation": ["evaluation", "benchmark", "metric", "accuracy", "评估", "指标", "准确"],
    "implementation": ["implementation", "deployment", "architecture", "system", "实现", "部署", "架构", "系统"],
}

METHOD_TAG_TERMS = {
    "literature review": ["review", "survey", "综述", "文献"],
    "case study": ["case study", "case", "案例"],
    "experiment": ["experiment", "trial", "benchmark", "实验", "测试"],
    "framework design": ["framework", "architecture", "model", "框架", "架构", "模型"],
    "policy analysis": ["policy", "standard", "guideline", "政策", "标准", "规范"],
}

USE_CASE_TAG_TERMS = {
    "research synthesis": ["synthesis", "review", "summary", "综述", "总结", "归纳"],
    "technical report": ["technical", "architecture", "implementation", "技术", "架构", "实现"],
    "business analysis": ["business", "operation", "management", "业务", "运营", "管理"],
    "risk control": ["risk", "governance", "compliance", "安全", "风险", "治理", "合规"],
    "decision support": ["decision", "recommendation", "planning", "决策", "建议", "规划"],
}

GENERAL_EVIDENCE_TERMS = [
    "evidence",
    "finding",
    "result",
    "method",
    "risk",
    "governance",
    "workflow",
    "retrieval",
    "knowledge",
    "automation",
    "evaluation",
    "limitation",
    "证据",
    "结论",
    "方法",
    "风险",
    "治理",
    "流程",
    "知识",
    "检索",
    "评估",
    "局限",
]


def build_source_cards(
    documents: list[Document],
    sections: list[OutlineSection],
    llm: LLMClient | None = None,
) -> list[SourceCard]:
    return [build_source_card(document, sections, llm=llm) for document in documents]


def flatten_source_card_evidence(cards: list[SourceCard]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for card in cards:
        for index, claim in enumerate(card.evidence_claims, start=1):
            rows.append(
                {
                    "ref_id": card.ref_id,
                    "title": card.title,
                    "claim_index": str(index),
                    "evidence_id": str(claim.get("evidence_id", "")),
                    "claim": str(claim.get("claim", "")),
                    "evidence_quote": str(claim.get("evidence_quote", "")),
                    "page_hint": str(claim.get("page_hint", "")),
                    "evidence_type": str(claim.get("evidence_type", "")),
                    "conclusion_strength": str(claim.get("conclusion_strength", "")),
                    "suggested_sections": "; ".join(str(item) for item in claim.get("suggested_sections", [])),
                    "experimental_model": str(claim.get("experimental_model", "")),
                    "outcome": str(claim.get("outcome", "")),
                    "limitation_note": str(claim.get("limitation_note", "")),
                    "source_window_id": str(claim.get("source_window_id", "")),
                }
            )
    return rows


def build_source_card(document: Document, sections: list[OutlineSection], llm: LLMClient | None = None) -> SourceCard:
    if llm is not None and not isinstance(llm, MockLLM):
        try:
            return model_source_card(document, sections, llm)
        except Exception:
            pass
    return heuristic_source_card(document, sections)


def heuristic_source_card(document: Document, sections: list[OutlineSection]) -> SourceCard:
    title = document.ref.title
    text = document.text[:35000]
    haystack = f"{title} {text}".lower()
    topic_tags = detect_terms(haystack, TOPIC_TAG_TERMS)
    method_tags = detect_terms(haystack, METHOD_TAG_TERMS)
    use_case_tags = detect_terms(haystack, USE_CASE_TAG_TERMS)
    key_findings = select_key_sentences(text, GENERAL_EVIDENCE_TERMS) or select_relevant_sentences(
        text,
        sections,
        limit=5,
    )
    limitations = select_key_sentences(text, ["limitation", "safety", "toxicity", "reactogenicity", "局限", "安全", "毒性"], limit=3)
    suggested_sections = suggest_sections(title + " " + clean_text(text, 3000), sections)
    evidence_claims = heuristic_evidence_claims(text, sections, limit=8)
    return SourceCard(
        ref_id=document.ref.ref_id,
        title=title,
        study_type=infer_study_type(haystack),
        topic_tags=topic_tags,
        method_tags=method_tags,
        use_case_tags=use_case_tags,
        key_findings=key_findings,
        limitations=limitations,
        suggested_sections=suggested_sections,
        evidence_claims=evidence_claims,
    )


def model_source_card(document: Document, sections: list[OutlineSection], llm: LLMClient) -> SourceCard:
    section_titles = writeable_section_titles(sections)
    source_windows = select_source_windows(document, sections, limit=14)
    system = (
        "You read one source file and turn it into a compact, source-grounded note for review drafting. "
        "The source may be an academic paper, markdown note, JSON record, table, report, or web text. "
        "Return strict JSON only. Be conservative and source-grounded."
    )
    user = (
        "请基于候选原文窗口生成一份资料摘要卡。输出 JSON 字段："
        "study_type, topic_tags, method_tags, use_case_tags, "
        "key_findings, limitations, suggested_sections, evidence_claims。\n"
        "所有 list 字段必须是字符串数组。suggested_sections 只能从给定大纲标题中选择，最多 6 个。\n"
        "evidence_claims 必须是对象数组，每个对象包含："
        "claim, evidence_quote, page_hint, evidence_type, conclusion_strength, suggested_sections, "
        "experimental_model, outcome, limitation_note, source_window_id。\n"
        "要求：\n"
        "1. evidence_claims 生成 6-10 条，优先覆盖背景、方法、技术路线、应用价值、风险治理、评价结果、局限性中可由原文支持的部分。\n"
        "2. evidence_quote 必须来自候选原文窗口，保持短句或短段，不要编造。\n"
        "3. conclusion_strength 只能是 direct, indirect, context。\n"
        "4. evidence_type 只能是 background, method, design, process, mechanism, implementation, application, governance, evaluation, safety, limitation。\n"
        "5. claim 必须比 evidence_quote 更简洁，不能超出原文证据强度。\n\n"
        f"ref_id: {document.ref.ref_id}\n"
        f"title: {document.ref.title}\n"
        f"outline_sections: {json.dumps(section_titles, ensure_ascii=False)}\n"
        "candidate_source_windows:\n"
        + "\n\n".join(
            f"[{item['source_window_id']}] page={item['page_hint']}\n{item['text']}"
            for item in source_windows
        )
    )
    data = parse_json_object(llm.complete(system, user, temperature=0.0))
    evidence_claims = normalize_evidence_claims(data.get("evidence_claims"), section_titles)
    key_findings = as_str_list(data.get("key_findings")) or [
        str(item.get("claim", "")).strip() for item in evidence_claims if item.get("claim")
    ][:5]
    return SourceCard(
        ref_id=document.ref.ref_id,
        title=document.ref.title,
        study_type=str(data.get("study_type", "research article")),
        topic_tags=as_str_list(data.get("topic_tags")) or ["unspecified"],
        method_tags=as_str_list(data.get("method_tags")) or ["unspecified"],
        use_case_tags=as_str_list(data.get("use_case_tags")) or ["unspecified"],
        key_findings=key_findings,
        limitations=as_str_list(data.get("limitations")),
        suggested_sections=[item for item in as_str_list(data.get("suggested_sections")) if item in section_titles]
        or suggest_sections(document.ref.title + " " + clean_text(document.text, 3000), sections),
        evidence_claims=evidence_claims or heuristic_evidence_claims(document.text, sections, limit=8),
    )


def writeable_section_titles(sections: list[OutlineSection]) -> list[str]:
    return [
        section.title
        for section in sections
        if section.level >= 2 and not section.title.strip().startswith("第")
    ]


def select_source_windows(document: Document, sections: list[OutlineSection], limit: int = 14) -> list[dict[str, str]]:
    query = f"{document.ref.title} " + " ".join(writeable_section_titles(sections))
    chunks = chunk_text(document.text, chunk_size=1800, overlap=260)
    ranked = sorted(
        enumerate(chunks, start=1),
        key=lambda item: token_score(query, item[1]),
        reverse=True,
    )
    selected: list[tuple[int, str]] = []
    if chunks:
        selected.append((1, chunks[0]))
    for item in ranked:
        if item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break
    windows = []
    for index, (chunk_index, text) in enumerate(selected, start=1):
        windows.append(
            {
                "source_window_id": f"W{index:03d}",
                "chunk_index": str(chunk_index),
                "page_hint": page_hint(text),
                "text": clean_text(text, 1800),
            }
        )
    return windows


def detect_terms(text: str, vocabulary: dict[str, list[str]]) -> list[str]:
    hits = []
    for label, terms in vocabulary.items():
        if any(term.lower() in text for term in terms):
            hits.append(label)
    return hits or ["unspecified"]


def infer_study_type(text: str) -> str:
    if "review" in text or "综述" in text:
        return "review"
    if "phase " in text or "clinical trial" in text:
        return "clinical"
    if "mice" in text or "mouse" in text or "murine" in text:
        return "animal experiment"
    if "in vitro" in text or "cell" in text:
        return "experimental"
    return "research article"


def select_key_sentences(text: str, terms: list[str], limit: int = 5) -> list[str]:
    sentences = split_sentences(clean_text(text[:35000]))
    ranked = sorted(sentences, key=lambda item: sum(term.lower() in item.lower() for term in terms), reverse=True)
    return [clean_text(item, 360) for item in ranked[:limit] if any(term.lower() in item.lower() for term in terms)]


def heuristic_evidence_claims(text: str, sections: list[OutlineSection], limit: int = 8) -> list[dict[str, Any]]:
    sentences = select_key_sentences(
        text,
        GENERAL_EVIDENCE_TERMS
        + [
            "design",
            "mechanism",
            "implementation",
            "application",
            "governance",
            "performance",
            "challenge",
            "设计",
            "机制",
            "实现",
            "应用",
            "效果",
            "挑战",
        ],
        limit=limit,
    )
    if not sentences:
        sentences = select_relevant_sentences(text, sections, limit=limit)
    claims: list[dict[str, Any]] = []
    for sentence in sentences:
        claims.append(
            {
                "claim": clean_text(sentence, 220),
                "evidence_quote": clean_text(sentence, 360),
                "page_hint": page_hint(sentence),
                "evidence_type": infer_evidence_type(sentence),
                "conclusion_strength": "context",
                "suggested_sections": suggest_sections(sentence, sections),
                "experimental_model": "",
                "outcome": "",
                "limitation_note": "",
                "source_window_id": "",
            }
        )
    return claims


def normalize_evidence_claims(value: Any, section_titles: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = clean_text(str(item.get("claim", "")), 320)
        evidence_quote = clean_text(str(item.get("evidence_quote", "")), 700)
        if not claim or not evidence_quote:
            continue
        suggested = [section for section in as_str_list(item.get("suggested_sections")) if section in section_titles]
        evidence_type = str(item.get("evidence_type", "")).strip()
        if evidence_type not in {
            "background",
            "design",
            "process",
            "mechanism",
            "application",
            "implementation",
            "governance",
            "evaluation",
            "safety",
            "limitation",
            "method",
        }:
            evidence_type = infer_evidence_type(f"{claim} {evidence_quote}")
        strength = str(item.get("conclusion_strength", "")).strip()
        if strength not in {"direct", "indirect", "context"}:
            strength = "context"
        normalized.append(
            {
                "evidence_id": "",
                "claim": claim,
                "evidence_quote": evidence_quote,
                "page_hint": str(item.get("page_hint", "")).strip(),
                "evidence_type": evidence_type,
                "conclusion_strength": strength,
                "suggested_sections": suggested,
                "experimental_model": clean_text(str(item.get("experimental_model", "")), 180),
                "outcome": clean_text(str(item.get("outcome", "")), 240),
                "limitation_note": clean_text(str(item.get("limitation_note", "")), 240),
                "source_window_id": clean_text(str(item.get("source_window_id", "")), 40),
            }
        )
    return normalized


def infer_evidence_type(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["background", "context", "现状", "背景", "需求"]):
        return "background"
    if any(term in lowered for term in ["safety", "toxicity", "biocompat", "安全", "毒性"]):
        return "safety"
    if any(term in lowered for term in ["risk", "governance", "compliance", "audit", "风险", "治理", "合规", "审计"]):
        return "governance"
    if any(term in lowered for term in ["evaluation", "metric", "benchmark", "accuracy", "评估", "指标", "准确"]):
        return "evaluation"
    if any(term in lowered for term in ["limitation", "challenge", "局限", "不足", "挑战"]):
        return "limitation"
    if any(term in lowered for term in ["implementation", "workflow", "pipeline", "deploy", "architecture", "流程", "架构", "部署"]):
        return "implementation"
    if any(term in lowered for term in ["method", "prepared", "synthesized", "制备", "方法"]):
        return "method"
    if any(term in lowered for term in ["mechanism", "pathway", "factor", "driver", "机制", "路径", "因素", "原因"]):
        return "mechanism"
    if any(term in lowered for term in ["process", "step", "operation", "control", "流程", "步骤", "过程", "控制"]):
        return "process"
    if any(term in lowered for term in ["design", "structure", "architecture", "framework", "model", "设计", "结构", "架构", "框架", "模型"]):
        return "design"
    if any(term in lowered for term in ["application", "scenario", "value", "use case", "应用", "场景", "价值"]):
        return "application"
    return "background"


def suggest_sections(text: str, sections: list[OutlineSection]) -> list[str]:
    ranked = sorted(sections, key=lambda section: token_score(section.title, text), reverse=True)
    return [section.title for section in ranked[:3]]


def select_relevant_sentences(text: str, sections: list[OutlineSection], limit: int = 5) -> list[str]:
    sentences = split_sentences(clean_text(text[:35000]))
    if not sentences:
        chunks = chunk_text(text, chunk_size=360, overlap=0)
        return [clean_text(item, 360) for item in chunks[:limit]]
    query = " ".join(section.title for section in sections)
    ranked = sorted(sentences, key=lambda item: token_score(query, item), reverse=True)
    return [clean_text(item, 360) for item in ranked[:limit]]


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


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

