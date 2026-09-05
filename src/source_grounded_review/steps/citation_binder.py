from __future__ import annotations

from collections import defaultdict

from source_grounded_review.core.models import ClaimAudit, Evidence


def build_citation_bindings(audits: list[ClaimAudit], evidence_rows: list[Evidence]) -> list[dict[str, object]]:
    evidence_by_id = {item.evidence_id: item for item in evidence_rows}
    rows: list[dict[str, object]] = []
    for audit in audits:
        if not audit.citation_ids:
            rows.append(
                {
                    "claim_id": audit.claim_id,
                    "section": audit.section,
                    "claim_text": audit.claim_text,
                    "citation_id": "",
                    "evidence_id": "",
                    "evidence_ref_id": "",
                    "page_hint": "",
                    "verdict": audit.verdict,
                    "relevance_score": round(audit.relevance_score, 4),
                    "binding_status": "needs_citation",
                    "evidence_text": "",
                }
            )
            continue
        evidence_by_ref: dict[str, list[Evidence]] = defaultdict(list)
        for evidence_id in audit.best_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence:
                evidence_by_ref[evidence.ref_id].append(evidence)
        for citation_id in audit.citation_ids:
            candidates = evidence_by_ref.get(citation_id, [])
            if not candidates:
                rows.append(
                    {
                        "claim_id": audit.claim_id,
                        "section": audit.section,
                        "claim_text": audit.claim_text,
                        "citation_id": citation_id,
                        "evidence_id": "",
                        "evidence_ref_id": "",
                        "page_hint": "",
                        "verdict": audit.verdict,
                        "relevance_score": round(audit.relevance_score, 4),
                        "binding_status": "no_matching_evidence",
                        "evidence_text": "",
                    }
                )
                continue
            for evidence in candidates[:2]:
                rows.append(
                    {
                        "claim_id": audit.claim_id,
                        "section": audit.section,
                        "claim_text": audit.claim_text,
                        "citation_id": citation_id,
                        "evidence_id": evidence.evidence_id,
                        "evidence_ref_id": evidence.ref_id,
                        "page_hint": evidence.page_hint,
                        "verdict": audit.verdict,
                        "relevance_score": round(audit.relevance_score, 4),
                        "binding_status": "bound",
                        "evidence_text": evidence.evidence_text,
                        "source_path": evidence.source_path,
                        "source_quote": evidence.source_quote,
                        "source_start": evidence.source_start,
                        "source_end": evidence.source_end,
                        "source_sha256": evidence.source_sha256,
                    }
                )
    methods = {audit.claim_id: audit.audit_method for audit in audits}
    for row in rows:
        row["audit_method"] = methods[str(row["claim_id"])]
    return rows

