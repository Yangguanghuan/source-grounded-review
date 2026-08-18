from __future__ import annotations

import re
from pathlib import Path

from source_grounded_review.core.models import Evidence, OutlineSection
from source_grounded_review.core.utils import token_score


DEFAULT_OUTLINE = [
    "研究背景与问题提出",
    "资料来源与分析方法",
    "核心技术路线与实现方式",
    "典型应用场景与业务价值",
    "风险问题与治理机制",
    "总结与未来方向",
]


def load_outline(path: Path | None) -> list[OutlineSection]:
    if path is None:
        return [
            OutlineSection(section_id=f"S{index:02d}", title=title, level=1, raw=title)
            for index, title in enumerate(DEFAULT_OUTLINE, start=1)
        ]
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections: list[OutlineSection] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        level = 1
        title = stripped
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
        else:
            match = re.match(r"^(\d+(?:\.\d+)*)[、.．\s]+(.+)$", stripped)
            if match:
                level = match.group(1).count(".") + 1
                title = match.group(2).strip()
        if is_reference_section(title):
            continue
        if title:
            sections.append(
                OutlineSection(section_id=f"S{len(sections) + 1:02d}", title=title, level=level, raw=stripped)
            )
    return sections or load_outline(None)


def is_reference_section(title: str) -> bool:
    compact = re.sub(r"\s+", "", title)
    return bool(re.search(r"参考文献|references$", compact, re.IGNORECASE))


def map_evidence_to_outline(sections: list[OutlineSection], evidence_rows: list[Evidence]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section in sections:
        ranked = sorted(
            evidence_rows,
            key=lambda item: token_score(section.title, f"{item.section} {item.evidence_text}") + item.score * 0.1,
            reverse=True,
        )
        top = ranked[:20]
        rows.append(
            {
                "section_id": section.section_id,
                "section_title": section.title,
                "evidence_count": len(top),
                "ref_count": len({item.ref_id for item in top}),
                "evidence_ids": "; ".join(item.evidence_id for item in top),
                "ref_ids": "; ".join(sorted({item.ref_id for item in top}, key=str)),
                "coverage": "sufficient" if len({item.ref_id for item in top}) >= 3 else "weak",
            }
        )
    return rows

