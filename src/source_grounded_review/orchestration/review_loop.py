from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypedDict

from source_grounded_review.steps.audit import audit_draft
from source_grounded_review.steps.assignment import assign_source_cards_to_outline
from source_grounded_review.steps.citation_binder import build_citation_bindings
from source_grounded_review.io.corpus import load_corpus, references_to_rows
from source_grounded_review.steps.coverage import build_coverage_critique, build_reference_usage
from source_grounded_review.steps.evidence import build_evidence_matrix
from source_grounded_review.steps.evidence_pack import build_section_evidence_packs, flatten_section_evidence_packs
from source_grounded_review.core.llm import LLMClient, build_llm
from source_grounded_review.core.models import (
    ClaimAudit,
    Document,
    Evidence,
    OutlineSection,
    SourceCard,
    SourceSectionAssignment,
    SectionEvidencePack,
)
from source_grounded_review.io.outline import load_outline, map_evidence_to_outline
from source_grounded_review.steps.source_reader import build_source_cards, flatten_source_card_evidence
from source_grounded_review.reporting.report import render_run_summary
from source_grounded_review.steps.selection import SectionSelectionConfig, select_section_evidence
from source_grounded_review.core.state_store import OutputRecord, SQLiteStateStore
from source_grounded_review.core.utils import clean_text, write_csv, write_json, write_text
from source_grounded_review.steps.writer import write_review_draft
from source_grounded_review.orchestration.task_planner import (
    TASK_TOOLS, execute_targeted_task, finish_standard_task, finish_task, plan_next_task, record_stop,
)


RISK_VERDICTS = {"misaligned", "overstated", "needs_more_citation", "unknown_citation"}
SOFT_REVIEW_VERDICTS = {"partially_supports"}
REVIEW_STEPS = [
    "plan_tasks",
    "retrieve_evidence",
    "revise_claim",
    "audit_section",
    "load_corpus",
    "build_source_cards",
    "build_evidence_matrix",
    "outline_mapper",
    "select_evidence",
    "build_section_packs",
    "expand_evidence",
    "write_draft",
    "audit_citations",
    "rewrite_draft",
    "coverage_critic",
    "human_review",
]

PLANNER_MODES = {"rules", "llm"}

class GraphRuntimeState(TypedDict):
    runtime: "EvidenceState"
    next_node: str


@dataclass
class ReviewLoopConfig:
    evidence_per_source: int = 10
    max_sections_per_source: int = 3
    max_refs_per_section: int = 6
    max_evidence_per_section: int = 10
    max_evidence_per_ref_section: int = 2
    use_model_selection: bool = False
    min_refs_per_section: int = 2
    min_evidence_per_section: int = 3
    max_revision_rounds: int = 2
    max_evidence_expansion_rounds: int = 2
    expansion_step_sections_per_source: int = 1
    expansion_step_refs_per_section: int = 2
    expansion_step_evidence_per_section: int = 4
    expansion_step_evidence_per_ref_section: int = 1
    max_flagged_claim_ratio: float = 0.25
    max_misaligned_claims: int = 0
    planner_mode: str = "rules"
    max_agent_tasks: int = 24
    max_retrieval_tasks: int = 4
    max_claim_revisions: int = 6


@dataclass
class StepDecision:
    step: int
    after_node: str
    action: str
    reason: str
    risk: dict[str, Any]
    planner_mode: str = "rules"
    planner_status: str = "rules"
    candidate_actions: str = ""
    planner_raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data["risk"].items():
            if isinstance(value, list):
                data["risk"][key] = "; ".join(str(item) for item in value[:20])
        if len(data.get("planner_raw_response", "")) > 1000:
            data["planner_raw_response"] = data["planner_raw_response"][:1000]
        return data


@dataclass(frozen=True)
class RouteOption:
    action: str
    reason: str


@dataclass
class EvidenceState:
    topic: str
    input_dir: Path
    output_dir: Path
    references_path: Path | None
    outline_file: Path | None
    max_sources: int | None
    llm: LLMClient
    config: ReviewLoopConfig
    store: SQLiteStateStore | None = field(default=None, repr=False)
    sections: list[OutlineSection] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    cards: list[SourceCard] = field(default_factory=list)
    evidence_rows: list[Evidence] = field(default_factory=list)
    assignments: list[SourceSectionAssignment] = field(default_factory=list)
    selected_evidence_by_section: dict[str, list[str]] = field(default_factory=dict)
    selection_rows: list[dict[str, object]] = field(default_factory=list)
    section_packs: list[SectionEvidencePack] = field(default_factory=list)
    outline_map: list[dict[str, object]] = field(default_factory=list)
    draft: str = ""
    audits: list[ClaimAudit] = field(default_factory=list)
    citation_bindings: list[dict[str, object]] = field(default_factory=list)
    usage_rows: list[dict[str, object]] = field(default_factory=list)
    unused_rows: list[dict[str, object]] = field(default_factory=list)
    coverage_critique_rows: list[dict[str, object]] = field(default_factory=list)
    coverage_critique_report: str = ""
    audit_feedback_by_section: dict[str, list[str]] = field(default_factory=dict)
    human_review_queue: list[dict[str, object]] = field(default_factory=list)
    step_decisions: list[StepDecision] = field(default_factory=list)
    node_trace: list[dict[str, object]] = field(default_factory=list)
    state_snapshots: list[dict[str, object]] = field(default_factory=list)
    revision_round: int = 0
    evidence_expansion_round: int = 0
    current_max_sections_per_source: int = 3
    current_max_refs_per_section: int = 6
    current_max_evidence_per_section: int = 10
    current_max_evidence_per_ref_section: int = 2
    graph_backend: str = "local_loop"
    active_task: dict[str, Any] = field(default_factory=dict)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    dirty_sections: list[str] = field(default_factory=list)
    evidence_dirty_sections: list[str] = field(default_factory=list)
    draft_audited: bool = False
    coverage_checked: bool = False
    _human_review_keys: set[str] = field(default_factory=set, repr=False)

    def snapshot(self, node: str) -> dict[str, Any]:
        risk = build_risk_summary(self)
        return {
            "node": node,
            "topic": self.topic,
            "documents": len(self.documents),
            "sections": len(self.sections),
            "source_cards": len(self.cards),
            "evidence_rows": len(self.evidence_rows),
            "assignments": len(self.assignments),
            "section_packs": len(self.section_packs),
            "draft_chars": len(self.draft),
            "audited_claims": len(self.audits),
            "revision_round": self.revision_round,
            "evidence_expansion_round": self.evidence_expansion_round,
            "current_max_sections_per_source": self.current_max_sections_per_source,
            "current_max_refs_per_section": self.current_max_refs_per_section,
            "current_max_evidence_per_section": self.current_max_evidence_per_section,
            "current_max_evidence_per_ref_section": self.current_max_evidence_per_ref_section,
            "graph_backend": self.graph_backend,
            "planner_mode": self.config.planner_mode,
            "human_review_items": len(self.human_review_queue),
            "risk": risk,
            "task_count": len(self.task_history),
            "active_task_id": self.active_task.get("task_id"),
            "dirty_sections": list(self.dirty_sections),
            "evidence_dirty_sections": list(self.evidence_dirty_sections),
        }


def run_review_loop(
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
    min_refs_per_section: int = 2,
    min_evidence_per_section: int = 3,
    max_revision_rounds: int = 2,
    max_evidence_expansion_rounds: int = 2,
    max_flagged_claim_ratio: float = 0.25,
    max_misaligned_claims: int = 0,
    graph_backend: str = "auto",
    planner_mode: str = "rules",
    max_agent_tasks: int = 24,
    max_retrieval_tasks: int = 4,
    max_claim_revisions: int = 6,
) -> dict[str, str]:
    output_path = Path(output_dir)
    planner_mode = planner_mode.strip().lower()
    if planner_mode not in PLANNER_MODES:
        raise ValueError("planner_mode must be one of: rules, llm")
    if not 1 <= max_agent_tasks <= 30 or max_retrieval_tasks < 0 or max_claim_revisions < 0:
        raise ValueError("Agent task budget must be 1-30; tool budgets must be nonnegative")
    config = ReviewLoopConfig(
        evidence_per_source=evidence_per_source,
        max_refs_per_section=max_refs_per_section,
        max_evidence_per_section=max_evidence_per_section,
        max_evidence_per_ref_section=max_evidence_per_ref_section,
        use_model_selection=use_model_selection,
        min_refs_per_section=min_refs_per_section,
        min_evidence_per_section=min_evidence_per_section,
        max_revision_rounds=max_revision_rounds,
        max_evidence_expansion_rounds=max_evidence_expansion_rounds,
        max_flagged_claim_ratio=max_flagged_claim_ratio,
        max_misaligned_claims=max_misaligned_claims,
        planner_mode=planner_mode,
        max_agent_tasks=max_agent_tasks,
        max_retrieval_tasks=max_retrieval_tasks,
        max_claim_revisions=max_claim_revisions,
    )
    state = EvidenceState(
        topic=topic,
        input_dir=Path(input_dir),
        output_dir=output_path,
        references_path=Path(references_csv) if references_csv else None,
        outline_file=Path(outline_path) if outline_path else None,
        max_sources=max_sources,
        llm=build_llm(provider),
        config=config,
        store=SQLiteStateStore(output_path / "state" / "evidence_state.sqlite"),
        current_max_sections_per_source=config.max_sections_per_source,
        current_max_refs_per_section=config.max_refs_per_section,
        current_max_evidence_per_section=config.max_evidence_per_section,
        current_max_evidence_per_ref_section=config.max_evidence_per_ref_section,
    )

    execute_review_loop(state, graph_backend=graph_backend)
    return persist_outputs(state)


def execute_review_loop(state: EvidenceState, *, graph_backend: str = "auto") -> None:
    requested = graph_backend.strip().lower()
    if requested not in {"auto", "langgraph", "local"}:
        raise ValueError("graph_backend must be one of: auto, langgraph, local")
    if requested == "local":
        state.graph_backend = "local_loop"
        execute_local_loop(state)
        return
    if langgraph_available():
        state.graph_backend = "langgraph"
        execute_langgraph_loop(state)
        return
    if requested == "langgraph":
        raise RuntimeError("The requested graph backend is not installed. Use graph_backend='local' or install the optional graph package.")
    state.graph_backend = "local_loop"
    execute_local_loop(state)


def execute_local_loop(state: EvidenceState) -> None:
    next_node = "load_corpus"
    step_guard = 0
    while next_node != "finalize":
        step_guard += 1
        if step_guard > 80:
            queue_human_review(
                state,
                {
                    "issue_type": "loop_guard_stop",
                    "severity": "critical",
                    "node": next_node,
                    "reason": "运行步骤超过 80 步，已停止以避免无限循环。",
                },
            )
            break
        run_node(state, next_node)
        next_node = route_after_node(state, next_node)


def execute_langgraph_loop(state: EvidenceState) -> None:
    with warnings.catch_warnings():
        ignore_langgraph_deprecation_warnings()
        from langgraph.graph import END, StateGraph

        graph = StateGraph(GraphRuntimeState)
        for node_name in REVIEW_STEPS:
            graph.add_node(node_name, make_langgraph_node(node_name))
        graph.set_entry_point("load_corpus")
        path_map = {node_name: node_name for node_name in REVIEW_STEPS}
        path_map["finalize"] = END
        for node_name in REVIEW_STEPS:
            graph.add_conditional_edges(node_name, route_langgraph_state, path_map)
        compiled = graph.compile()
        result = compiled.invoke({"runtime": state, "next_node": "load_corpus"}, {"recursion_limit": 100})
    returned_state = result.get("runtime")
    if isinstance(returned_state, EvidenceState) and returned_state is not state:
        state.__dict__.update(returned_state.__dict__)


def make_langgraph_node(node_name: str) -> Callable[[GraphRuntimeState], GraphRuntimeState]:
    def _node(graph_state: GraphRuntimeState) -> GraphRuntimeState:
        state = graph_state["runtime"]
        run_node(state, node_name)
        next_node = route_after_node(state, node_name)
        return {"runtime": state, "next_node": next_node}

    return _node


def route_langgraph_state(graph_state: GraphRuntimeState) -> str:
    next_node = graph_state["next_node"]
    return "finalize" if next_node == "finalize" else next_node


def langgraph_available() -> bool:
    with warnings.catch_warnings():
        ignore_langgraph_deprecation_warnings()
        try:
            import langgraph  # noqa: F401
        except ImportError:
            return False
    return True


def ignore_langgraph_deprecation_warnings() -> None:
    try:
        from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    except ImportError:
        warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
    else:
        warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


def run_node(state: EvidenceState, node: str) -> None:
    try:
        _run_node(state, node)
    except Exception as exc:
        if state.config.planner_mode != "llm" or state.active_task.get("action") != node:
            raise
        finish_task(state, {"criterion_met": False, "error": f"{type(exc).__name__}: {exc}"[:1200]}, status="failed")
        record_snapshot(state, node)


def _run_node(state: EvidenceState, node: str) -> None:
    if node == "plan_tasks":
        plan_next_task(state)
    elif node in TASK_TOOLS:
        execute_targeted_task(state, node)
    elif node == "load_corpus":
        state.sections = load_outline(state.outline_file)
        state.documents = load_corpus(state.references_path, state.input_dir, max_sources=state.max_sources)
    elif node == "build_source_cards":
        state.cards = build_source_cards(state.documents, state.sections, llm=state.llm)
    elif node == "build_evidence_matrix":
        state.evidence_rows = build_evidence_matrix(
            state.documents,
            state.cards,
            state.sections,
            state.topic,
            per_source=state.config.evidence_per_source,
        )
    elif node == "outline_mapper":
        state.assignments = assign_source_cards_to_outline(
            state.cards,
            state.sections,
            state.evidence_rows,
            max_sections_per_source=state.current_max_sections_per_source,
        )
        state.outline_map = map_evidence_to_outline(state.sections, state.evidence_rows)
    elif node == "select_evidence":
        state.selected_evidence_by_section, state.selection_rows = select_section_evidence(
            state.sections,
            state.assignments,
            state.evidence_rows,
            state.cards,
            SectionSelectionConfig(
                max_refs_per_section=state.current_max_refs_per_section,
                max_evidence_per_section=state.current_max_evidence_per_section,
                max_evidence_per_ref_section=state.current_max_evidence_per_ref_section,
                use_model=state.config.use_model_selection,
            ),
            llm=state.llm,
        )
    elif node == "build_section_packs":
        state.section_packs = build_section_evidence_packs(
            state.sections,
            state.assignments,
            state.evidence_rows,
            state.cards,
            selected_evidence_by_section=state.selected_evidence_by_section,
        )
    elif node == "expand_evidence":
        expand_evidence_budget(state)
    elif node == "write_draft":
        state.draft = write_review_draft(
            state.topic,
            state.sections,
            state.section_packs,
            state.llm,
            audit_feedback_by_section=state.audit_feedback_by_section,
        )
    elif node == "audit_citations":
        state.audits = audit_draft(state.draft, state.evidence_rows, llm=state.llm)
        state.citation_bindings = build_citation_bindings(state.audits, state.evidence_rows)
    elif node == "rewrite_draft":
        state.revision_round += 1
        state.audit_feedback_by_section = build_audit_feedback(state.audits)
        state.draft = write_review_draft(
            state.topic,
            state.sections,
            state.section_packs,
            state.llm,
            audit_feedback_by_section=state.audit_feedback_by_section,
        )
    elif node == "coverage_critic":
        state.usage_rows, state.unused_rows = build_reference_usage(
            state.documents,
            state.cards,
            state.evidence_rows,
            state.draft,
            state.audits,
            state.assignments,
        )
        state.coverage_critique_rows, state.coverage_critique_report = build_coverage_critique(
            state.documents,
            state.assignments,
            state.section_packs,
            state.usage_rows,
            min_refs_per_section=state.config.min_refs_per_section,
            min_evidence_per_section=state.config.min_evidence_per_section,
        )
    elif node == "human_review":
        if state.active_task.get("action") == "human_review":
            record_stop(state)
        if not state.human_review_queue:
            queue_human_review(
                state,
                {
                    "issue_type": "manual_review_gate",
                    "severity": "medium",
                    "reason": "运行过程进入人工复核，但没有更具体的失败项。",
                },
            )
    else:
        raise ValueError(f"Unknown review step: {node}")

    if state.active_task.get("action") == node and node not in TASK_TOOLS:
        finish_standard_task(state, node)
    record_snapshot(state, node)


def route_after_node(state: EvidenceState, after_node: str) -> str:
    if state.config.planner_mode == "llm" and after_node == "plan_tasks":
        task = state.active_task
        action, reason = task["action"], task["reason"]
        planner_status, planner_raw = task["planner_status"], json.dumps(task, ensure_ascii=False)
        options = [RouteOption(a, "Task tool") for a in task["available_actions"]]
        if action == "finalize":
            finish_task(state, {"criterion_met": True})
    elif state.config.planner_mode == "llm" and (
        after_node == "build_section_packs" or after_node in TASK_TOOLS
        or (state.active_task and after_node in {"write_draft", "audit_citations", "coverage_critic"})
    ):
        action, reason = "plan_tasks", "Observe tool results and plan the next concrete task."
        planner_status, planner_raw = "task_observation", ""
        options = [RouteOption(action, reason)]
    else:
        options = route_options_after_node(state, after_node)
        action, reason, planner_status, planner_raw = select_route(state, after_node, options)
    apply_route_side_effects(state, after_node, action)
    risk = build_risk_summary(state)

    decision = StepDecision(
        step=len(state.step_decisions) + 1,
        after_node=after_node,
        action=action,
        reason=reason,
        risk=risk,
        planner_mode=state.config.planner_mode,
        planner_status=planner_status,
        candidate_actions=", ".join(option.action for option in options),
        planner_raw_response=planner_raw,
    )
    state.step_decisions.append(decision)
    if state.store:
        state.store.add_step_decision(
            step=decision.step,
            after_node=after_node,
            action=action,
            reason=reason,
            risk=risk,
        )
    return action


def select_route(
    state: EvidenceState,
    after_node: str,
    options: list[RouteOption],
) -> tuple[str, str, str, str]:
    if not options:
        return "finalize", "No route option was available, so the run is finalized.", "empty_options", ""
    option = options[0]
    status = "rules" if state.config.planner_mode == "rules" else "preparation"
    return option.action, option.reason, status, ""


def route_options_after_node(state: EvidenceState, after_node: str) -> list[RouteOption]:
    if after_node == "load_corpus":
        if not state.documents:
            return [RouteOption("human_review", "No readable source material was loaded.")]
        return [RouteOption("build_source_cards", "Sources were loaded; build source cards next.")]
    if after_node == "build_source_cards":
        if not state.cards:
            return [RouteOption("human_review", "No source cards were produced, so evidence extraction cannot continue.")]
        return [RouteOption("build_evidence_matrix", "Source cards are ready; extract citable evidence next.")]
    if after_node == "build_evidence_matrix":
        if not state.evidence_rows:
            return [RouteOption("human_review", "The evidence matrix is empty.")]
        return [RouteOption("outline_mapper", "Evidence rows are ready; map them to the outline.")]
    if after_node == "outline_mapper":
        if not state.assignments:
            return [RouteOption("human_review", "No source was assigned to any outline section.")]
        return [RouteOption("select_evidence", "Outline mapping is ready; select section evidence.")]
    if after_node == "select_evidence":
        return [RouteOption("build_section_packs", "Section evidence selection is ready; build section packs.")]
    if after_node == "build_section_packs":
        gaps = section_evidence_gaps(state)
        if gaps and can_expand_evidence(state):
            return [
                RouteOption("expand_evidence", f"{len(gaps)} sections have weak evidence coverage; broaden candidate evidence."),
                RouteOption("write_draft", "Continue drafting with current evidence and record weak sections for review."),
                RouteOption("human_review", "Stop automatic drafting because section evidence coverage is weak."),
            ]
        if gaps:
            return [
                RouteOption("write_draft", "Evidence expansion is exhausted; continue drafting and record weak sections for review."),
                RouteOption("human_review", "Stop automatic drafting because section evidence remains weak."),
            ]
        return [RouteOption("write_draft", "Section evidence packs are ready for drafting.")]
    if after_node == "expand_evidence":
        return [RouteOption("outline_mapper", "Evidence budgets were expanded; remap evidence to the outline.")]
    if after_node == "write_draft":
        return [RouteOption("audit_citations", "Draft text is ready; audit citation support next.")]
    if after_node == "audit_citations":
        risk = build_risk_summary(state)
        has_soft_review = int(risk["soft_review_claims"]) > 0
        if citation_risk_exceeds_threshold(state):
            if state.revision_round < state.config.max_revision_rounds:
                return [
                    RouteOption("rewrite_draft", "Citation audit found high-risk claims; rewrite using audit feedback."),
                    RouteOption("coverage_critic", "Continue to coverage review and keep citation risks for manual review."),
                    RouteOption("human_review", "Stop automatic work because citation risk is too high."),
                ]
            return [
                RouteOption("coverage_critic", "Citation risk remains after maximum rewrite rounds; continue and mark risks for review."),
                RouteOption("human_review", "Stop automatic work because citation risk remains unresolved."),
            ]
        if has_soft_review and state.revision_round < state.config.max_revision_rounds:
            return [
                RouteOption("coverage_critic", "Citation audit risk is within threshold; continue to coverage review."),
                RouteOption("rewrite_draft", "Some claims are only partially supported; rewrite before coverage review."),
            ]
        return [RouteOption("coverage_critic", "Citation audit risk is within threshold; continue to coverage review.")]
    if after_node == "rewrite_draft":
        return [RouteOption("audit_citations", "The draft was rewritten; audit citations again.")]
    if after_node == "coverage_critic":
        gaps = section_evidence_gaps(state)
        if gaps and can_expand_evidence(state):
            return [
                RouteOption("expand_evidence", "Coverage review found weak sections; expand evidence and remap."),
                RouteOption("finalize", "Finalize with weak sections recorded for review."),
                RouteOption("human_review", "Stop automatic work because coverage is weak."),
            ]
        if gaps or state.unused_rows:
            return [
                RouteOption("finalize", "Finalize with coverage notes and manual review items."),
                RouteOption("human_review", "Stop automatic work and keep coverage issues for manual review."),
            ]
        return [RouteOption("finalize", "Generation, citation audit, and coverage review are complete.")]
    if after_node == "human_review":
        return [RouteOption("finalize", "Manual review notes were queued; finalize the run.")]
    return [RouteOption("finalize", "All required checks completed.")]


def apply_route_side_effects(state: EvidenceState, after_node: str, action: str) -> None:
    if after_node == "load_corpus" and action == "human_review" and not state.documents:
        queue_human_review(
            state,
            {
                "issue_type": "empty_corpus",
                "severity": "critical",
                "reason": "No readable PDF/TXT/MD/JSON/CSV/HTML source material was loaded.",
            },
        )
    elif after_node == "build_source_cards" and action == "human_review" and not state.cards:
        queue_human_review(
            state,
            {
                "issue_type": "source_card_failure",
                "severity": "critical",
                "reason": "No source cards were produced, so evidence extraction cannot continue.",
            },
        )
    elif after_node == "build_evidence_matrix" and action == "human_review" and not state.evidence_rows:
        queue_human_review(
            state,
            {
                "issue_type": "no_evidence_rows",
                "severity": "critical",
                "reason": "No evidence rows were produced.",
            },
        )
    elif after_node == "outline_mapper" and action == "human_review" and not state.assignments:
        queue_human_review(
            state,
            {
                "issue_type": "outline_mapping_failure",
                "severity": "critical",
                "reason": "No source was assigned to any outline section.",
            },
        )

    if after_node == "build_section_packs" and action in {"write_draft", "human_review"}:
        gaps = section_evidence_gaps(state)
        if gaps:
            queue_section_gaps(state, gaps, "weak_section_after_selection")
    if after_node == "audit_citations" and action in {"coverage_critic", "human_review"}:
        if citation_risk_exceeds_threshold(state):
            queue_citation_risks(state)
    if after_node == "coverage_critic" and action in {"finalize", "human_review"}:
        gaps = section_evidence_gaps(state)
        if gaps:
            queue_section_gaps(state, gaps, "weak_section_final")
        if state.unused_rows:
            queue_unused_sources(state)
    if after_node == "human_review" and not state.human_review_queue:
        queue_human_review(
            state,
            {
                "issue_type": "manual_review_gate",
                "severity": "medium",
                "reason": "The run entered manual review without a more specific issue.",
            },
        )


def expand_evidence_budget(state: EvidenceState) -> None:
    config = state.config
    state.evidence_expansion_round += 1
    state.current_max_sections_per_source += config.expansion_step_sections_per_source
    state.current_max_refs_per_section += config.expansion_step_refs_per_section
    state.current_max_evidence_per_section += config.expansion_step_evidence_per_section
    state.current_max_evidence_per_ref_section += config.expansion_step_evidence_per_ref_section


def can_expand_evidence(state: EvidenceState) -> bool:
    return state.evidence_expansion_round < state.config.max_evidence_expansion_rounds


def citation_risk_exceeds_threshold(state: EvidenceState) -> bool:
    if not state.audits:
        return False
    risk = build_risk_summary(state)
    if int(risk["misaligned_claims"]) > state.config.max_misaligned_claims:
        return True
    return float(risk["flagged_claim_ratio"]) > state.config.max_flagged_claim_ratio


def build_risk_summary(state: EvidenceState) -> dict[str, Any]:
    verdict_counts = Counter(audit.verdict for audit in state.audits)
    flagged = sum(verdict_counts[verdict] for verdict in RISK_VERDICTS)
    soft_review = sum(verdict_counts[verdict] for verdict in SOFT_REVIEW_VERDICTS)
    total = len(state.audits)
    gaps = section_evidence_gaps(state)
    return {
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "audited_claims": total,
        "flagged_claims": flagged,
        "soft_review_claims": soft_review,
        "flagged_claim_ratio": round(flagged / total, 4) if total else 0.0,
        "misaligned_claims": verdict_counts.get("misaligned", 0),
        "overstated_claims": verdict_counts.get("overstated", 0),
        "needs_more_citation_claims": verdict_counts.get("needs_more_citation", 0),
        "weak_sections": len(gaps),
        "weak_section_titles": [item["section_title"] for item in gaps],
        "unused_sources": len(state.unused_rows),
    }


def section_evidence_gaps(state: EvidenceState) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    for pack in state.section_packs:
        ref_count = len(pack.assigned_ref_ids)
        evidence_count = len(pack.evidence_ids)
        if ref_count < state.config.min_refs_per_section or evidence_count < state.config.min_evidence_per_section:
            gaps.append(
                {
                    "section_id": pack.section_id,
                    "section_title": pack.section_title,
                    "assigned_ref_count": ref_count,
                    "evidence_count": evidence_count,
                    "min_refs_per_section": state.config.min_refs_per_section,
                    "min_evidence_per_section": state.config.min_evidence_per_section,
                }
            )
    return gaps


def build_audit_feedback(audits: list[ClaimAudit]) -> dict[str, list[str]]:
    feedback: dict[str, list[str]] = defaultdict(list)
    for audit in audits:
        if audit.verdict not in RISK_VERDICTS | SOFT_REVIEW_VERDICTS:
            continue
        feedback[audit.section].append(
            clean_text(
                f"{audit.verdict}: {audit.claim_text} | citations={', '.join(audit.citation_ids) or 'none'} | {audit.rationale}",
                500,
            )
        )
    return dict(feedback)


def queue_human_review(state: EvidenceState, item: dict[str, object]) -> None:
    key = "|".join(
        str(item.get(part, ""))
        for part in ["issue_type", "section_id", "section_title", "claim_id", "ref_id", "reason"]
    )
    if key in state._human_review_keys:
        return
    state._human_review_keys.add(key)
    state.human_review_queue.append(item)
    if state.store:
        state.store.add_human_review_item(item)


def queue_section_gaps(state: EvidenceState, gaps: list[dict[str, object]], issue_type: str) -> None:
    for gap in gaps:
        queue_human_review(
            state,
            {
                "issue_type": issue_type,
                "severity": "medium",
                **gap,
                "reason": "章节证据覆盖低于设定阈值，需要人工确认是否降低覆盖要求、补充资料或调整大纲。",
            },
        )


def queue_citation_risks(state: EvidenceState) -> None:
    for audit in state.audits:
        if audit.verdict not in RISK_VERDICTS:
            continue
        queue_human_review(
            state,
            {
                "issue_type": "citation_audit_failed_after_rewrites",
                "severity": "high",
                "claim_id": audit.claim_id,
                "section_title": audit.section,
                "claim_text": audit.claim_text,
                "citation_ids": "; ".join(audit.citation_ids),
                "best_evidence_ids": "; ".join(audit.best_evidence_ids),
                "verdict": audit.verdict,
                "relevance_score": round(audit.relevance_score, 4),
                "reason": audit.rationale,
            },
        )


def queue_unused_sources(state: EvidenceState) -> None:
    for row in state.unused_rows:
        queue_human_review(
            state,
            {
                "issue_type": "reference_not_used_in_draft",
                "severity": "low",
                "ref_id": row.get("ref_id", ""),
                "title": row.get("title", ""),
                "document_path": row.get("document_path", ""),
                "reason": "该资料已读取并生成摘要卡，但最终正文未引用；如要求全库高参与度，需要人工复核。",
            },
        )


def record_snapshot(state: EvidenceState, node: str) -> None:
    snapshot = state.snapshot(node)
    state.state_snapshots.append(snapshot)
    state.node_trace.append(
        {
            "step": len(state.node_trace) + 1,
            "node": node,
            "documents": snapshot["documents"],
            "source_cards": snapshot["source_cards"],
            "evidence_rows": snapshot["evidence_rows"],
            "assignments": snapshot["assignments"],
            "audited_claims": snapshot["audited_claims"],
            "revision_round": snapshot["revision_round"],
            "evidence_expansion_round": snapshot["evidence_expansion_round"],
            "human_review_items": snapshot["human_review_items"],
        }
    )
    if state.store:
        state.store.add_snapshot(len(state.state_snapshots), node, snapshot)


def persist_outputs(state: EvidenceState) -> dict[str, str]:
    paths: dict[str, str] = {}
    records: list[OutputRecord] = []

    def record(key: str, path: str, output_type: str, producer: str, description: str = "") -> None:
        paths[key] = path
        item = OutputRecord(key, path, output_type, producer, description)
        records.append(item)
        if state.store:
            state.store.add_output(item)

    output_dir = state.output_dir
    if state.config.planner_mode == "llm":
        record("task_history_json", write_json(output_dir / "trace" / "task_history.json", state.task_history),
               "task_history", "task_planner", "Task goals, parameters, criteria, observations and revisions.")
        record("task_report", write_text(output_dir / "report" / "task_report.md", render_task_report(state)),
               "task_report", "task_planner")
    record(
        "references_resolved_csv",
        write_csv(output_dir / "corpus" / "references_resolved.csv", references_to_rows(state.documents)),
        "corpus_table",
        "source_loader",
        "用户提供资料与可读全文路径的解析结果。",
    )
    record(
        "source_cards_json",
        write_json(output_dir / "corpus" / "source_cards.json", [card.to_dict() for card in state.cards]),
        "source_card",
        "source_reader",
        "每份资料的结构化阅读卡。",
    )
    record(
        "source_cards_csv",
        write_csv(output_dir / "corpus" / "source_cards.csv", [card.to_dict() for card in state.cards]),
        "source_card_table",
        "source_reader",
    )
    record(
        "source_card_evidence_csv",
        write_csv(output_dir / "corpus" / "source_card_evidence.csv", flatten_source_card_evidence(state.cards)),
        "source_card_claim_table",
        "source_reader",
    )
    evidence_notes = build_evidence_notes(state.evidence_rows, state.assignments)
    record(
        "evidence_notes_json",
        write_json(output_dir / "records" / "evidence_notes.json", evidence_notes),
        "evidence_note",
        "evidence_builder",
        "带页码、来源、适用章节和边界说明的证据片段。",
    )
    record(
        "evidence_notes_csv",
        write_csv(output_dir / "records" / "evidence_notes.csv", flatten_evidence_notes(evidence_notes)),
        "evidence_note_table",
        "evidence_builder",
    )
    record(
        "evidence_matrix_json",
        write_json(output_dir / "evidence" / "evidence_matrix.json", [item.to_dict() for item in state.evidence_rows]),
        "evidence_matrix",
        "evidence_builder",
    )
    record(
        "evidence_matrix_csv",
        write_csv(output_dir / "evidence" / "evidence_matrix.csv", [item.to_dict() for item in state.evidence_rows]),
        "evidence_matrix_table",
        "evidence_builder",
    )
    record(
        "source_outline_assignments_json",
        write_json(output_dir / "outline" / "source_outline_assignments.json", [item.to_dict() for item in state.assignments]),
        "outline_assignment",
        "outline_matcher",
    )
    record(
        "source_outline_assignments_csv",
        write_csv(output_dir / "outline" / "source_outline_assignments.csv", [item.to_dict() for item in state.assignments]),
        "outline_assignment_table",
        "outline_matcher",
    )
    record(
        "outline_evidence_map_json",
        write_json(output_dir / "outline" / "outline_evidence_map.json", state.outline_map),
        "outline_evidence_map",
        "outline_matcher",
    )
    record(
        "outline_evidence_map_csv",
        write_csv(output_dir / "outline" / "outline_evidence_map.csv", state.outline_map),
        "outline_evidence_map_table",
        "outline_matcher",
    )
    record(
        "section_evidence_packs_json",
        write_json(output_dir / "packs" / "section_evidence_packs.json", [pack.to_dict() for pack in state.section_packs]),
        "section_evidence_pack",
        "section_picker",
    )
    record(
        "section_evidence_packs_csv",
        write_csv(output_dir / "packs" / "section_evidence_packs.csv", flatten_section_evidence_packs(state.section_packs)),
        "section_evidence_pack_table",
        "section_picker",
    )
    record(
        "section_evidence_selection_csv",
        write_csv(output_dir / "packs" / "section_evidence_selection.csv", state.selection_rows),
        "section_selection_trace",
        "section_picker",
    )
    record("review_draft", write_text(output_dir / "draft" / "review_draft.md", state.draft), "draft", "writer")
    record(
        "citation_bindings_csv",
        write_csv(output_dir / "bindings" / "citation_bindings.csv", state.citation_bindings),
        "citation_binding_table",
        "citation_binder",
    )
    record(
        "citation_bindings_json",
        write_json(output_dir / "bindings" / "citation_bindings.json", state.citation_bindings),
        "citation_binding",
        "citation_binder",
    )
    record(
        "citation_audit_json",
        write_json(output_dir / "audit" / "citation_audit_results.json", [item.to_dict() for item in state.audits]),
        "citation_audit",
        "citation_checker",
    )
    record(
        "citation_audit_csv",
        write_csv(output_dir / "audit" / "citation_audit_results.csv", [item.to_dict() for item in state.audits]),
        "citation_audit_table",
        "citation_checker",
    )
    record(
        "reference_usage_csv",
        write_csv(output_dir / "coverage" / "reference_usage_matrix.csv", state.usage_rows),
        "coverage_table",
        "coverage_checker",
    )
    record(
        "unused_sources_csv",
        write_csv(output_dir / "coverage" / "unused_sources.csv", state.unused_rows),
        "coverage_table",
        "coverage_checker",
    )
    record(
        "coverage_critique_csv",
        write_csv(output_dir / "coverage" / "coverage_critique.csv", state.coverage_critique_rows),
        "coverage_table",
        "coverage_checker",
    )
    record(
        "coverage_critique_report",
        write_text(output_dir / "coverage" / "coverage_report.md", state.coverage_critique_report),
        "coverage_report",
        "coverage_checker",
    )
    record(
        "step_decisions_json",
        write_json(output_dir / "trace" / "step_decisions.json", [item.to_dict() for item in state.step_decisions]),
        "step_trace",
        "review_loop",
    )
    record(
        "step_decisions_csv",
        write_csv(output_dir / "trace" / "step_decisions.csv", [item.to_dict() for item in state.step_decisions]),
        "step_trace_table",
        "review_loop",
    )
    record(
        "node_trace_csv",
        write_csv(output_dir / "trace" / "step_trace.csv", state.node_trace),
        "node_trace_table",
        "review_loop",
    )
    record(
        "evidence_state_snapshots_json",
        write_json(output_dir / "state" / "evidence_state_snapshots.json", state.state_snapshots),
        "state_snapshot",
        "state_store",
    )
    record(
        "evidence_state_final_json",
        write_json(output_dir / "state" / "evidence_state_final.json", state.snapshot("finalize")),
        "state_snapshot",
        "state_store",
    )
    record(
        "manual_review_queue_csv",
        write_csv(output_dir / "manual_review" / "manual_review_queue.csv", state.human_review_queue),
        "manual_review_queue",
        "manual_review",
    )
    record(
        "manual_review_queue_md",
        write_text(output_dir / "manual_review" / "manual_review_notes.md", render_manual_review_queue(state.human_review_queue)),
        "manual_review_queue_report",
        "manual_review",
    )
    record(
        "evidence_state_sqlite",
        str(state.store.path if state.store else output_dir / "state" / "evidence_state.sqlite"),
        "sqlite_state_store",
        "state_store",
        "Step decisions, state snapshots, manual review items, and output metadata.",
    )
    record(
        "run_summary",
        write_text(
            output_dir / "report" / "run_summary.md",
            render_run_summary(
                topic=state.topic,
                documents=state.documents,
                evidence_rows=state.evidence_rows,
                audits=state.audits,
                output_paths=paths,
            ),
        ),
        "run_summary",
        "Reporter",
    )
    record(
        "review_loop_summary",
        write_text(output_dir / "report" / "review_loop_summary.md", render_loop_summary(state, paths)),
        "loop_summary",
        "Reporter",
    )
    record(
        "output_manifest_json",
        write_json(output_dir / "records" / "output_manifest.json", [item.to_dict() for item in records]),
        "output_manifest",
        "output_manifest",
    )
    record(
        "output_manifest_csv",
        write_csv(output_dir / "records" / "output_manifest.csv", [item.to_dict() for item in records]),
        "output_manifest_table",
        "output_manifest",
    )
    return paths


def build_evidence_notes(
    evidence_rows: list[Evidence],
    assignments: list[SourceSectionAssignment],
) -> list[dict[str, object]]:
    sections_by_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for assignment in assignments:
        for evidence_id in assignment.evidence_ids:
            sections_by_evidence[evidence_id].append(
                {
                    "section_id": assignment.section_id,
                    "section_title": assignment.section_title,
                    "role": assignment.role,
                    "support_strength": assignment.support_strength,
                }
            )
    notes: list[dict[str, object]] = []
    for item in evidence_rows:
        notes.append(
            {
                "note_id": f"N-{item.evidence_id}",
                "evidence_id": item.evidence_id,
                "ref_id": item.ref_id,
                "title": item.title,
                "source_path": item.source_path,
                "page_hint": item.page_hint,
                "confidence_score": round(item.score, 4),
                "evidence_text": item.evidence_text,
                "source_quote": item.source_quote,
                "source_context": item.source_context,
                "source_start": item.source_start,
                "source_end": item.source_end,
                "source_sha256": item.source_sha256,
                "applicable_sections": sections_by_evidence.get(item.evidence_id, []),
                "validity_boundary": (
                    "Only supports claims directly entailed by evidence_text and the cited source. "
                    "Do not use this note for broader claims without additional evidence."
                ),
            }
        )
    return notes


def render_task_report(state: EvidenceState) -> str:
    lines = ["# Task Report", "", f"Topic: {state.topic}", ""]
    for task in state.task_history:
        lines.extend([f"## {task['task_id']}: {task['action']}", "",
                      f"Goal: {task['goal']}", "", f"Decision: {task['reason']}", "",
                      f"Status: {task['status']}", "", f"Completion criterion: `{task['success_criterion']}`", "",
                      "```json", json.dumps({"parameters": task['parameters'],
                                              "outcome": task.get('outcome', {}),
                                              "remaining_plan": task.get('remaining_plan', [])},
                                             ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def flatten_evidence_notes(notes: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for note in notes:
        sections = note.get("applicable_sections", [])
        section_text = ""
        if isinstance(sections, list):
            section_text = "; ".join(str(item.get("section_title", "")) for item in sections if isinstance(item, dict))
        rows.append(
            {
                "note_id": note.get("note_id", ""),
                "evidence_id": note.get("evidence_id", ""),
                "ref_id": note.get("ref_id", ""),
                "title": note.get("title", ""),
                "source_path": note.get("source_path", ""),
                "page_hint": note.get("page_hint", ""),
                "confidence_score": note.get("confidence_score", ""),
                "applicable_sections": section_text,
                "validity_boundary": note.get("validity_boundary", ""),
                "evidence_text": note.get("evidence_text", ""),
            }
        )
    return rows


def render_manual_review_queue(items: list[dict[str, object]]) -> str:
    lines = ["# Manual Review Notes", ""]
    if not items:
        lines.append("当前没有需要人工复核的条目。")
        return "\n".join(lines)
    for index, item in enumerate(items, start=1):
        lines.append(f"## HR{index:04d} {item.get('issue_type', 'review_item')}")
        lines.append("")
        lines.append(f"- Severity: {item.get('severity', '')}")
        for key, value in item.items():
            if key in {"issue_type", "severity"}:
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def render_loop_summary(state: EvidenceState, paths: dict[str, str]) -> str:
    risk = build_risk_summary(state)
    lines = [
        "# Review Loop Summary",
        "",
        f"- Topic: {state.topic}",
        f"- Documents read: {len(state.documents)}",
        f"- Source cards: {len(state.cards)}",
        f"- Evidence rows: {len(state.evidence_rows)}",
        f"- Step decisions: {len(state.step_decisions)}",
        f"- Graph backend: {state.graph_backend}",
        f"- Planner mode: {state.config.planner_mode}",
        f"- Evidence expansion rounds: {state.evidence_expansion_round}",
        f"- Draft revision rounds: {state.revision_round}",
        f"- Manual review items: {len(state.human_review_queue)}",
        "",
        "## Final Risk State",
        "",
        f"- Audited claims: {risk['audited_claims']}",
        f"- Flagged claims: {risk['flagged_claims']}",
        f"- Flagged claim ratio: {risk['flagged_claim_ratio']}",
        f"- Weak sections: {risk['weak_sections']}",
        f"- Unused references: {risk['unused_sources']}",
        "",
        "## Step Decisions",
        "",
    ]
    for decision in state.step_decisions:
        lines.append(
            f"- after `{decision.after_node}` -> `{decision.action}` "
            f"({decision.planner_status}): {decision.reason}"
        )
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            f"- Draft: `{paths.get('review_draft', '')}`",
            f"- Step decisions: `{paths.get('step_decisions_csv', '')}`",
            f"- Evidence state snapshots: `{paths.get('evidence_state_snapshots_json', '')}`",
            f"- Evidence notes: `{paths.get('evidence_notes_csv', '')}`",
            f"- Citation check: `{paths.get('citation_audit_csv', '')}`",
            f"- Manual review notes: `{paths.get('manual_review_queue_md', '')}`",
            f"- SQLite state store: `{paths.get('evidence_state_sqlite', '')}`",
            "",
            "## Interpretation",
            "",
            "This run keeps a small shared state and can loop back to broaden evidence, rewrite the draft, or leave notes for manual review. "
            "A non-empty manual review file does not mean the run failed; it marks places that should be checked before sharing the draft.",
            "",
        ]
    )
    return "\n".join(lines)

