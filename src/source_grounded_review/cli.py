from __future__ import annotations

import argparse
from pathlib import Path

from source_grounded_review.orchestration.pipeline import run_pipeline
from source_grounded_review.orchestration.review_loop import run_review_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft and check a source-grounded review from local files.")
    parser.add_argument("--topic", required=True, help="Review or research-report topic.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", dest="input_dir", help="Directory containing user-provided PDF/TXT/MD/JSON/CSV/HTML files.")
    input_group.add_argument("--paper-dir", dest="input_dir", help=argparse.SUPPRESS)
    parser.add_argument("--references", default=None, help="Optional CSV reference list.")
    parser.add_argument("--outline", default=None, help="Optional Markdown outline.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--provider", choices=["mock", "deepseek"], default="mock", help="LLM provider.")
    parser.add_argument("--max-sources", dest="max_sources", type=int, default=None, help="Optional source limit for small trial runs.")
    parser.add_argument("--max-papers", dest="max_sources", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--evidence-per-source", dest="evidence_per_source", type=int, default=10, help="Evidence claims/chunks retained per source.")
    parser.add_argument("--evidence-per-paper", dest="evidence_per_source", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-refs-per-section", type=int, default=6, help="Selected sources retained per outline section.")
    parser.add_argument("--max-evidence-per-section", type=int, default=10, help="Selected evidence rows retained per outline section.")
    parser.add_argument(
        "--max-evidence-per-ref-section",
        type=int,
        default=2,
        help="Selected evidence rows retained from one source in one outline section.",
    )
    parser.add_argument(
        "--model-select-evidence",
        dest="use_model_selection",
        action="store_true",
        help="Use the configured model to choose final evidence from rule-ranked section candidates.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Run the simpler one-pass pipeline without coverage/rewrite feedback loops.",
    )
    parser.add_argument("--max-revision-rounds", type=int, default=2, help="Maximum audit-driven draft rewrite rounds.")
    parser.add_argument(
        "--max-evidence-expansion-rounds",
        type=int,
        default=2,
        help="Maximum times to broaden section evidence when coverage is weak.",
    )
    parser.add_argument(
        "--min-refs-per-section",
        type=int,
        default=2,
        help="Minimum selected sources per section before a coverage warning is raised.",
    )
    parser.add_argument(
        "--min-evidence-per-section",
        type=int,
        default=3,
        help="Minimum selected evidence rows per section before a coverage warning is raised.",
    )
    parser.add_argument(
        "--max-flagged-claim-ratio",
        type=float,
        default=0.25,
        help="Flagged-claim ratio that triggers a rewrite.",
    )
    parser.add_argument(
        "--max-misaligned-claims",
        type=int,
        default=0,
        help="Misaligned-claim count that triggers a rewrite.",
    )
    parser.add_argument(
        "--graph-backend",
        choices=["auto", "langgraph", "local"],
        default="local",
        help="Execution backend for the review loop. auto chooses the graph runner when available and otherwise uses the local runner.",
    )
    parser.add_argument(
        "--planner",
        choices=["rules", "llm"],
        default="rules",
        help="Use rules or an LLM task planner with targeted retrieval, claim revision and re-audit.",
    )
    parser.add_argument("--max-agent-tasks", type=int, default=24, help="LLM task budget (1-30).")
    parser.add_argument("--max-retrieval-tasks", type=int, default=4, help="Maximum original-source retrieval tasks.")
    parser.add_argument("--max-claim-revisions", type=int, default=6, help="Maximum targeted claim revisions.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    common_args = {
        "topic": args.topic,
        "input_dir": Path(args.input_dir),
        "output_dir": Path(args.output_dir),
        "references_csv": Path(args.references) if args.references else None,
        "outline_path": Path(args.outline) if args.outline else None,
        "provider": args.provider,
        "max_sources": args.max_sources,
        "evidence_per_source": args.evidence_per_source,
        "max_refs_per_section": args.max_refs_per_section,
        "max_evidence_per_section": args.max_evidence_per_section,
        "max_evidence_per_ref_section": args.max_evidence_per_ref_section,
        "use_model_selection": args.use_model_selection,
    }
    use_review_loop = not args.linear
    if use_review_loop:
        paths = run_review_loop(
            **common_args,
            max_revision_rounds=args.max_revision_rounds,
            max_evidence_expansion_rounds=args.max_evidence_expansion_rounds,
            min_refs_per_section=args.min_refs_per_section,
            min_evidence_per_section=args.min_evidence_per_section,
            max_flagged_claim_ratio=args.max_flagged_claim_ratio,
            max_misaligned_claims=args.max_misaligned_claims,
            graph_backend=args.graph_backend,
            planner_mode=args.planner,
            max_agent_tasks=args.max_agent_tasks,
            max_retrieval_tasks=args.max_retrieval_tasks,
            max_claim_revisions=args.max_claim_revisions,
        )
    else:
        paths = run_pipeline(**common_args)
    print("Review completed.")
    print(f"Draft: {paths['review_draft']}")
    print(f"Audit CSV: {paths['citation_audit_csv']}")
    print(f"Summary: {paths['run_summary']}")
    if use_review_loop:
        print(f"Step decisions: {paths['step_decisions_csv']}")
        print(f"Manual review notes: {paths['manual_review_queue_md']}")


if __name__ == "__main__":
    main()

