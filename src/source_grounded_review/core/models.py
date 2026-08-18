from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Reference:
    ref_id: str
    title: str
    authors: str = ""
    year: str = ""
    source: str = ""
    doi: str = ""
    keywords: str = ""
    link: str = ""
    document_path: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    ref: Reference
    text: str


@dataclass
class SourceCard:
    ref_id: str
    title: str
    study_type: str
    topic_tags: list[str]
    method_tags: list[str]
    use_case_tags: list[str]
    key_findings: list[str]
    limitations: list[str]
    suggested_sections: list[str]
    evidence_claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    evidence_id: str
    ref_id: str
    title: str
    section: str
    page_hint: str
    score: float
    evidence_text: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(self.score, 4)
        return data


@dataclass
class OutlineSection:
    section_id: str
    title: str
    level: int
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceSectionAssignment:
    ref_id: str
    title: str
    section_id: str
    section_title: str
    role: str
    support_strength: str
    evidence_ids: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_ids"] = "; ".join(self.evidence_ids)
        return data


@dataclass
class SectionEvidencePack:
    section_id: str
    section_title: str
    assigned_ref_ids: list[str]
    evidence_ids: list[str]
    source_summaries: list[dict[str, Any]]
    evidence: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimAudit:
    claim_id: str
    section: str
    claim_text: str
    citation_ids: list[str]
    verdict: str
    best_evidence_ids: list[str]
    relevance_score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation_ids"] = "; ".join(self.citation_ids)
        data["best_evidence_ids"] = "; ".join(self.best_evidence_ids)
        data["relevance_score"] = round(self.relevance_score, 4)
        return data
