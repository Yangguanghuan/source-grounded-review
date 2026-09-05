from __future__ import annotations

from source_grounded_review.core.models import Document, Evidence, OutlineSection, SourceCard
from source_grounded_review.core.utils import chunk_text, clean_text, page_hint, token_score


def build_evidence_matrix(
    documents: list[Document],
    cards: list[SourceCard],
    sections: list[OutlineSection],
    topic: str,
    *,
    per_source: int = 5,
) -> list[Evidence]:
    cards_by_id = {card.ref_id: card for card in cards}
    evidence_rows: list[Evidence] = []
    for doc in documents:
        card = cards_by_id.get(doc.ref.ref_id)
        query = build_query(topic, sections, card)
        claim_rows = evidence_from_card_claims(doc, card, sections, query, per_source=per_source)
        if claim_rows:
            for row in claim_rows:
                row.evidence_id = f"E{len(evidence_rows) + 1:06d}"
                set_card_claim_evidence_id(card, row)
                evidence_rows.append(row)
            continue
        chunks = chunk_text(doc.text)
        ranked = sorted(chunks, key=lambda chunk: token_score(query, chunk), reverse=True)
        for chunk in ranked[:per_source]:
            best_section = best_matching_section(chunk, sections)
            evidence_rows.append(
                Evidence(
                    evidence_id=f"E{len(evidence_rows) + 1:06d}",
                    ref_id=doc.ref.ref_id,
                    title=doc.ref.title,
                    section=best_section.title,
                    page_hint=page_hint(chunk),
                    score=token_score(query, chunk),
                    evidence_text=clean_text(chunk, 1200),
                    source_path=doc.ref.document_path,
                    source_quote=clean_text(chunk, 1200),
                )
            )
    return evidence_rows


def evidence_from_card_claims(
    document: Document,
    card: SourceCard | None,
    sections: list[OutlineSection],
    query: str,
    *,
    per_source: int,
) -> list[Evidence]:
    if card is None or not card.evidence_claims:
        return []
    rows: list[Evidence] = []
    for claim in card.evidence_claims[:per_source]:
        claim_text = str(claim.get("claim", "")).strip()
        quote = str(claim.get("evidence_quote", "")).strip()
        if not claim_text or not quote:
            continue
        section = section_for_claim(claim, sections, quote)
        evidence_text = clean_text(f"claim: {claim_text} | source_quote: {quote}", 1200)
        rows.append(
            Evidence(
                evidence_id="",
                ref_id=document.ref.ref_id,
                title=document.ref.title,
                section=section.title,
                page_hint=str(claim.get("page_hint", "")).strip() or page_hint(quote),
                score=token_score(query, evidence_text),
                evidence_text=evidence_text,
                source_path=document.ref.document_path,
                source_quote=quote,
            )
        )
    return rows


def section_for_claim(claim: dict[str, object], sections: list[OutlineSection], text: str) -> OutlineSection:
    titles = claim.get("suggested_sections", [])
    if isinstance(titles, list):
        for title in titles:
            for section in sections:
                if section.title == str(title):
                    return section
    return best_matching_section(text, sections)


def set_card_claim_evidence_id(card: SourceCard | None, row: Evidence) -> None:
    if card is None:
        return
    for claim in card.evidence_claims:
        quote = str(claim.get("evidence_quote", "")).strip()
        if quote and quote[:80] in row.evidence_text:
            claim["evidence_id"] = row.evidence_id
            return


def build_query(topic: str, sections: list[OutlineSection], card: SourceCard | None) -> str:
    parts = [topic, " ".join(section.title for section in sections)]
    if card:
        parts.extend(card.topic_tags)
        parts.extend(card.method_tags)
        parts.extend(card.use_case_tags)
        parts.extend(card.key_findings[:2])
    return " ".join(parts)


def best_matching_section(text: str, sections: list[OutlineSection]) -> OutlineSection:
    return sorted(sections, key=lambda section: token_score(section.title, text), reverse=True)[0]

