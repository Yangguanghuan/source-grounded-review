from __future__ import annotations

import re

from source_grounded_review.core.llm import LLMClient, MockLLM
from source_grounded_review.core.models import OutlineSection, SectionEvidencePack
from source_grounded_review.core.utils import clean_text


CITATION_RE = re.compile(r"\[([A-Za-z]?\d+(?:\s*[,，、]\s*[A-Za-z]?\d+)*)\]")


def write_review_draft(
    topic: str,
    sections: list[OutlineSection],
    section_packs: list[SectionEvidencePack],
    llm: LLMClient,
    audit_feedback_by_section: dict[str, list[str]] | None = None,
) -> str:
    packs_by_section = {pack.section_id: pack for pack in section_packs}
    lines = [f"# {topic}", ""]
    for section in sections:
        pack = packs_by_section.get(section.section_id)
        feedback = (audit_feedback_by_section or {}).get(section.title, [])
        lines.append(f"## {section.title}")
        lines.append("")
        if pack is None:
            lines.append("本节当前没有分配到证据包，建议补充资料或重新执行大纲匹配。")
        elif not pack.evidence:
            lines.append("本节当前缺少可用证据，建议补充相关资料或调整资料-大纲分配。")
        elif isinstance(llm, MockLLM) and feedback:
            lines.append(conservative_section(pack))
        elif isinstance(llm, MockLLM):
            lines.append(mock_section(pack))
        else:
            lines.append(model_section(topic, pack, llm, audit_feedback=feedback))
        lines.append("")
    lines.append("## 证据链说明")
    lines.append("")
    lines.append(
        "本文由资料摘要卡、大纲匹配结果和章节证据包约束生成。"
        "每个数字引用均应在 `bindings/citation_bindings.csv` 和 `audit/citation_audit_results.csv` 中反查。"
    )
    return "\n".join(lines)


def mock_section(pack: SectionEvidencePack) -> str:
    if not pack.evidence:
        return "本节当前缺少可用证据，建议补充相关资料。"
    sentences: list[str] = []
    for summary in pack.source_summaries[:5]:
        ref_id = str(summary.get("ref_id", ""))
        role = str(summary.get("role", "supporting_evidence"))
        strength = str(summary.get("support_strength", "context"))
        finding = clean_text(str(summary.get("key_findings", "")), 120)
        if not finding and pack.evidence:
            finding = clean_text(str(pack.evidence[0].get("evidence_text", "")), 120)
        sentences.append(f"在“{pack.section_title}”中，文献 [{ref_id}] 可作为{role}证据（{strength}）：{finding}。")
    refs = ",".join(pack.assigned_ref_ids[: min(len(pack.assigned_ref_ids), 6)])
    sentences.append(f"总体来看，该方向需要综合比较材料参数、免疫通路、应用场景和安全边界[{refs}]。")
    return "\n\n".join(sentences)


def model_section(
    topic: str,
    pack: SectionEvidencePack,
    llm: LLMClient,
    audit_feedback: list[str] | None = None,
) -> str:
    allowed_refs = sorted(
        {str(ref_id) for ref_id in pack.assigned_ref_ids}
        | {str(item.get("ref_id", "")) for item in pack.evidence if item.get("ref_id")},
        key=str,
    )
    source_text = "\n".join(
        "- ref_id={ref_id}; role={role}; strength={support_strength}; reason={reason}; key_findings={key_findings}".format(
            **summary
        )
        for summary in pack.source_summaries
    )
    evidence_text = "\n".join(
        "- evidence_id={evidence_id}; ref_id={ref_id}; page={page_hint}; text={evidence_text}".format(**item)
        for item in pack.evidence
    )
    system = (
        "You write a source-grounded review draft. "
        "Write in Chinese. Use only the provided evidence. "
        "Every sentence containing factual content must cite one or more allowed ref_id values in square brackets, e.g. [33]. "
        "Prefer citing sources assigned to this section. "
        "Do not cite sources outside the provided evidence. "
        "Do not use external knowledge. Do not invent citations. Do not output a heading. "
        "If the evidence only supports a narrow conclusion, write the narrow conclusion."
    )
    feedback_text = ""
    if audit_feedback:
        feedback_text = (
            "\n上一轮引用检查对本节标记了以下问题。请删除无法支持的句子，"
            "或改写为证据能够直接支持的更窄表述：\n"
            + "\n".join(f"- {item}" for item in audit_feedback[:8])
            + "\n"
        )
    user = (
        f"综述主题：{topic}\n"
        f"本节标题：{pack.section_title}\n\n"
        "唯一允许使用的引用编号如下，正文中不得出现其他编号：\n"
        f"{', '.join(f'[{ref_id}]' for ref_id in allowed_refs)}\n\n"
        "本节已分配资料如下：\n"
        f"{source_text}\n\n"
        f"{feedback_text}"
        "本节可用原文证据如下。请写 2-4 段中文综述正文，必须严格基于这些证据：\n"
        f"{evidence_text}"
    )
    first = llm.complete(system, user, temperature=0.2)
    if citations_are_allowed(first, allowed_refs):
        return first
    repair_prompt = (
        f"下面正文出现了不在允许列表中的引用编号，或包含未被证据支持的泛化表述。\n"
        f"允许引用编号：{', '.join(f'[{ref_id}]' for ref_id in allowed_refs)}\n"
        "请删除无法由证据支持的句子，并重写为 2-4 段中文正文。"
        "每个实质性判断必须引用允许列表中的编号，不得出现其他编号，不得输出标题。\n\n"
        f"待修正文：\n{first}\n\n"
        f"可用证据：\n{evidence_text}"
    )
    repaired = llm.complete(system, repair_prompt, temperature=0.0)
    if citations_are_allowed(repaired, allowed_refs):
        return repaired
    return conservative_section(pack)


def citations_are_allowed(text: str, allowed_refs: list[str]) -> bool:
    allowed = set(allowed_refs)
    seen = extract_citation_ids(text)
    return bool(seen) and all(item in allowed for item in seen)


def extract_citation_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in CITATION_RE.finditer(text):
        raw = match.group(1).replace("，", ",").replace("、", ",")
        ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return ids


def conservative_section(pack: SectionEvidencePack) -> str:
    sentences: list[str] = []
    used_refs: set[str] = set()
    for item in pack.evidence[:6]:
        ref_id = str(item.get("ref_id", ""))
        evidence = clean_evidence_sentence(str(item.get("evidence_text", "")))
        if not ref_id or not evidence:
            continue
        used_refs.add(ref_id)
        sentences.append(f"已有证据显示，{evidence}[{ref_id}]。")
    if not sentences:
        return "本节当前缺少可用证据，建议补充相关资料或调整资料-大纲分配。"
    refs = ",".join(sorted(used_refs, key=str))
    sentences.append(f"因此，本节只能基于上述证据进行有限概括，后续仍需补充更多资料以提高论证覆盖度[{refs}]。")
    return "\n\n".join(sentences)


def clean_evidence_sentence(text: str) -> str:
    raw = clean_text(text, 500)
    if raw.startswith("claim:"):
        raw = raw.removeprefix("claim:").split("| source_quote:", 1)[0]
    raw = re.sub(r"^\s*#+\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = raw.strip(" 。.!?？;；|")
    return clean_text(raw, 180)

