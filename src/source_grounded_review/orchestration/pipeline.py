from __future__ import annotations

from pathlib import Path

from source_grounded_review.steps.audit import audit_draft
from source_grounded_review.steps.assignment import assign_source_cards_to_outline
from source_grounded_review.steps.citation_binder import build_citation_bindings
from source_grounded_review.io.corpus import load_corpus, references_to_rows
from source_grounded_review.steps.coverage import build_coverage_critique, build_reference_usage
from source_grounded_review.steps.evidence import build_evidence_matrix
from source_grounded_review.steps.evidence_pack import build_section_evidence_packs, flatten_section_evidence_packs
from source_grounded_review.core.llm import build_llm
from source_grounded_review.io.outline import load_outline, map_evidence_to_outline
from source_grounded_review.steps.source_reader import build_source_cards, flatten_source_card_evidence
from source_grounded_review.reporting.report import render_run_summary
from source_grounded_review.steps.selection import SectionSelectionConfig, select_section_evidence
from source_grounded_review.core.utils import write_csv, write_json, write_text
from source_grounded_review.steps.writer import write_review_draft


def run_pipeline(
    *,
    topic: str,
    input_dir: str | Path,
    output_dir: str | Path,
    references_csv: str | Path | None = None,
    outline_path: str | Path | None = None,
    provider: str = "mock",
    max_sources: int | None = None,
    evidence_per_source: int = 10,
    max_refs_per_section: int = 6,
    max_evidence_per_section: int = 10,
    max_evidence_per_ref_section: int = 2,
    use_model_selection: bool = False,
) -> dict[str, str]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    references_path = Path(references_csv) if references_csv else None
    outline_file = Path(outline_path) if outline_path else None

    llm = build_llm(provider)
    sections = load_outline(outline_file)
    documents = load_corpus(references_path, input_dir, max_sources=max_sources)
    cards = build_source_cards(documents, sections, llm=llm)
    evidence_rows = build_evidence_matrix(documents, cards, sections, topic, per_source=evidence_per_source)
    assignments = assign_source_cards_to_outline(cards, sections, evidence_rows)
    selected_evidence_by_section, selection_rows = select_section_evidence(
        sections,
        assignments,
        evidence_rows,
        cards,
        SectionSelectionConfig(
            max_refs_per_section=max_refs_per_section,
            max_evidence_per_section=max_evidence_per_section,
            max_evidence_per_ref_section=max_evidence_per_ref_section,
            use_model=use_model_selection,
        ),
        llm=llm,
    )
    section_packs = build_section_evidence_packs(
        sections,
        assignments,
        evidence_rows,
        cards,
        selected_evidence_by_section=selected_evidence_by_section,
    )
    outline_map = map_evidence_to_outline(sections, evidence_rows)
    draft = write_review_draft(topic, sections, section_packs, llm)
    audits = audit_draft(draft, evidence_rows, llm=llm)
    citation_bindings = build_citation_bindings(audits, evidence_rows)
    usage_rows, unused_rows = build_reference_usage(documents, cards, evidence_rows, draft, audits, assignments)
    coverage_critique_rows, coverage_critique_report = build_coverage_critique(
        documents,
        assignments,
        section_packs,
        usage_rows,
    )

    paths: dict[str, str] = {}
    paths["references_resolved_csv"] = write_csv(output_dir / "corpus" / "references_resolved.csv", references_to_rows(documents))
    paths["source_cards_json"] = write_json(output_dir / "corpus" / "source_cards.json", [card.to_dict() for card in cards])
    paths["source_cards_csv"] = write_csv(output_dir / "corpus" / "source_cards.csv", [card.to_dict() for card in cards])
    paths["source_card_evidence_csv"] = write_csv(
        output_dir / "corpus" / "source_card_evidence.csv",
        flatten_source_card_evidence(cards),
    )
    paths["evidence_matrix_json"] = write_json(
        output_dir / "evidence" / "evidence_matrix.json",
        [item.to_dict() for item in evidence_rows],
    )
    paths["evidence_matrix_csv"] = write_csv(
        output_dir / "evidence" / "evidence_matrix.csv",
        [item.to_dict() for item in evidence_rows],
    )
    paths["source_outline_assignments_json"] = write_json(
        output_dir / "outline" / "source_outline_assignments.json",
        [item.to_dict() for item in assignments],
    )
    paths["source_outline_assignments_csv"] = write_csv(
        output_dir / "outline" / "source_outline_assignments.csv",
        [item.to_dict() for item in assignments],
    )
    paths["outline_evidence_map_json"] = write_json(output_dir / "outline" / "outline_evidence_map.json", outline_map)
    paths["outline_evidence_map_csv"] = write_csv(output_dir / "outline" / "outline_evidence_map.csv", outline_map)
    paths["section_evidence_packs_json"] = write_json(
        output_dir / "packs" / "section_evidence_packs.json",
        [pack.to_dict() for pack in section_packs],
    )
    paths["section_evidence_packs_csv"] = write_csv(
        output_dir / "packs" / "section_evidence_packs.csv",
        flatten_section_evidence_packs(section_packs),
    )
    paths["section_evidence_selection_csv"] = write_csv(
        output_dir / "packs" / "section_evidence_selection.csv",
        selection_rows,
    )
    paths["review_draft"] = write_text(output_dir / "draft" / "review_draft.md", draft)
    paths["citation_bindings_csv"] = write_csv(output_dir / "bindings" / "citation_bindings.csv", citation_bindings)
    paths["citation_bindings_json"] = write_json(output_dir / "bindings" / "citation_bindings.json", citation_bindings)
    paths["citation_audit_json"] = write_json(
        output_dir / "audit" / "citation_audit_results.json",
        [item.to_dict() for item in audits],
    )
    paths["citation_audit_csv"] = write_csv(
        output_dir / "audit" / "citation_audit_results.csv",
        [item.to_dict() for item in audits],
    )
    paths["reference_usage_csv"] = write_csv(output_dir / "coverage" / "reference_usage_matrix.csv", usage_rows)
    paths["unused_sources_csv"] = write_csv(output_dir / "coverage" / "unused_sources.csv", unused_rows)
    paths["coverage_critique_csv"] = write_csv(output_dir / "coverage" / "coverage_critique.csv", coverage_critique_rows)
    paths["coverage_critique_report"] = write_text(
        output_dir / "coverage" / "coverage_critic_report.md",
        coverage_critique_report,
    )
    paths["run_summary"] = write_text(
        output_dir / "report" / "run_summary.md",
        render_run_summary(
            topic=topic,
            documents=documents,
            evidence_rows=evidence_rows,
            audits=audits,
            output_paths=paths,
        ),
    )
    return paths

