from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from source_grounded_review.core.llm import LLMClient, MockLLM
from source_grounded_review.core.models import Document, Evidence, OutlineSection, Reference, SectionEvidencePack, SourceCard
from source_grounded_review.core.state_store import SQLiteStateStore
from source_grounded_review.orchestration.review_loop import (
    EvidenceState, ReviewLoopConfig, REVIEW_STEPS, langgraph_available, make_langgraph_node,
    persist_outputs, route_after_node, route_langgraph_state, run_node, GraphRuntimeState,
)
from source_grounded_review.orchestration.task_planner import (
    CRITERIA, legal_actions, plan_next_task, validate_task,
)
from source_grounded_review.steps.audit import audit_draft
from source_grounded_review.steps.targeted import retrieve_targeted_evidence, revise_targeted_claim, source_windows
from source_grounded_review.steps.source_reader import model_source_card
from source_grounded_review.steps.citation_binder import build_citation_bindings


QUOTE = "Deployment cost was not measured in this benchmark."
BROAD = "The optimized runtime reduces deployment costs by 40%[P001]."
NARROW = "Deployment cost was not measured in this benchmark[P001]."
OTHER = "The evaluation used a fixed hardware configuration throughout the benchmark[P001]."


class ResponseLLM(LLMClient):
    def __init__(self, response):
        self.response = response

    def complete(self, system, user, *, temperature=0.2):
        return self.response(system, user) if callable(self.response) else self.response


def task(action, **parameters):
    return {"action": action, "parameters": parameters, "goal": "Resolve the unsupported cost claim",
            "reason": "Use the observed evidence and audit result", "success_criterion": CRITERIA[action],
            "remaining_plan": ["Reassess the result before the next task"]}


def make_state(root, llm):
    sections = [OutlineSection("S1", "Deployment costs", 1, "# Deployment costs"),
                OutlineSection("S2", "Evaluation setup", 1, "# Evaluation setup")]
    text = "[PAGE 1]\nThe evaluation used a fixed hardware configuration throughout the benchmark.\n[PAGE 2]\n" + QUOTE
    doc = Document(Reference("P001", "Runtime benchmark", document_path=""), text)
    evidence = Evidence("E000001", "P001", doc.ref.title, sections[1].title, "1", 1.0,
                        "The evaluation used a fixed hardware configuration throughout the benchmark.", "")
    state = EvidenceState("Runtime evaluation", Path(root), Path(root) / "output", None, None, None, llm,
                          ReviewLoopConfig(planner_mode="llm", min_refs_per_section=1, min_evidence_per_section=1),
                          store=SQLiteStateStore(Path(root) / "state.sqlite"))
    state.sections = sections
    state.documents = [doc]
    state.cards = [SourceCard("P001", doc.ref.title, "benchmark", [], [], [], [evidence.evidence_text], [], [])]
    state.evidence_rows = [evidence]
    state.section_packs = [SectionEvidencePack(s.section_id, s.title, ["P001"], [evidence.evidence_id], [],
                                             [evidence.to_dict()]) for s in sections]
    state.draft = f"# Runtime evaluation\n\n## Deployment costs\n\n{BROAD}\n\n## Evaluation setup\n\n{OTHER}\n"
    return state


class ScenarioLLM(LLMClient):
    """A deterministic tool-protocol fixture, not a model-quality evaluation."""
    def __init__(self, retrieve=True):
        self.retrieve = retrieve

    def complete(self, system, user, *, temperature=0.2):
        if system.startswith("You plan research-report tasks"):
            state = json.loads(user)["state"]
            available = json.loads(user)["available_tools"]
            if not state["draft_audited"]:
                answer = task("audit_citations")
            elif state["dirty_sections"]:
                answer = task("audit_section", section_id=state["dirty_sections"][0])
            elif state["unresolved_claims"]:
                claim = state["unresolved_claims"][0]
                retrieved = any(t["action"] == "retrieve_evidence" for t in state["recent_tasks"])
                if self.retrieve and not retrieved:
                    answer = task("retrieve_evidence", section_id="S1", claim_id=claim["claim_id"],
                                  source_ids=["P001"], query="deployment cost measurements benchmark limitations")
                else:
                    catalog = state["evidence_by_section"]["S1"]
                    answer = task("revise_claim", section_id="S1", claim_id=claim["claim_id"],
                                  evidence_ids=[catalog[-1]["evidence_id"]],
                                  instruction="Replace the cost reduction claim with the measured limitation")
            elif "coverage_critic" in available:
                answer = task("coverage_critic")
            else:
                answer = task("finalize")
            return json.dumps(answer)
        if system.startswith("Extract relevant evidence"):
            windows = json.loads(user)["windows"]
            window = next(w for w in windows if QUOTE in w["text"])
            return json.dumps({"matches": [{"window_id": window["window_id"], "quote": QUOTE}], "reason": "No cost measurement"})
        if system.startswith("Revise only"):
            return json.dumps({"replacement": NARROW if self.retrieve else OTHER, "reason": "Narrow to measured evidence"})
        if system.startswith("You are a conservative citation"):
            claim = user.split("claim: ", 1)[1].split("\ncitations:", 1)[0]
            return json.dumps({"verdict": "overstated" if "costs by 40%" in claim else "supports",
                               "rationale": "Cost savings were not measured" if "costs by 40%" in claim else "Explicit source statement"})
        raise AssertionError(f"Unexpected model call: {system[:80]}")


class TaskPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_retrieve_preserves_page_offsets_and_rejects_invented_quote(self):
        state = make_state(self.temp.name, ScenarioLLM())
        rows, outcome = retrieve_targeted_evidence(state.documents, state.evidence_rows, source_ids=["P001"],
            query="deployment cost", section_title="Deployment costs", llm=state.llm)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.page_hint, "2")
        self.assertEqual(state.documents[0].text[row.source_start:row.source_end], QUOTE)
        self.assertEqual(len(row.source_sha256), 64)
        fake = ResponseLLM(json.dumps({"matches": [{"window_id": "W0", "quote": "This fabricated quote does not exist in the original document."}]}))
        rows, outcome = retrieve_targeted_evidence(state.documents, [], source_ids=["P001"],
            query="deployment cost", section_title="Deployment costs", llm=fake)
        self.assertEqual(rows, [])
        self.assertEqual(outcome["rejected_quotes"], 1)

    def test_windows_never_cross_pages(self):
        windows = source_windows("P001", "[PAGE 1]" + "a" * 4000 + "[PAGE 2]" + "b" * 4000)
        self.assertTrue(all(not ("a" in w.text and "b" in w.text) for w in windows))
        self.assertEqual({w.page for w in windows}, {"1", "2"})

    def test_retrieval_rereads_beyond_initial_corpus_limit(self):
        state = make_state(self.temp.name, ScenarioLLM())
        path = Path(self.temp.name) / "long.md"
        path.write_text("irrelevant " * 13000 + "\n[PAGE 9]\n" + QUOTE, encoding="utf-8")
        state.documents[0].ref.document_path = str(path)
        state.documents[0].text = "truncated initial corpus"
        rows, _ = retrieve_targeted_evidence(state.documents, [], source_ids=["P001"], query="deployment cost",
                                            section_title="Deployment costs", llm=state.llm)
        self.assertEqual(rows[0].page_hint, "9")

    def test_local_revision_preserves_other_section_exactly(self):
        state = make_state(self.temp.name, ScenarioLLM())
        before = state.draft
        result, _ = revise_targeted_claim(before, title="Deployment costs", claim=BROAD, instruction="Narrow",
                                          evidence=state.evidence_rows, llm=state.llm)
        self.assertEqual(result, before.replace(BROAD, NARROW))

    def test_invalid_citation_and_ambiguous_edit_rejected(self):
        state = make_state(self.temp.name, ScenarioLLM())
        for replacement in ("A completely unsupported factual conclusion[P999].", "## Another heading[P001]"):
            with self.assertRaises(ValueError):
                revise_targeted_claim(state.draft, title="Deployment costs", claim=BROAD, instruction="Narrow",
                    evidence=state.evidence_rows, llm=ResponseLLM(json.dumps({"replacement": replacement})))
        with self.assertRaises(ValueError):
            revise_targeted_claim(state.draft.replace(BROAD, BROAD + "\n" + BROAD), title="Deployment costs",
                claim=BROAD, instruction="Narrow", evidence=state.evidence_rows, llm=state.llm)

    def test_invalid_task_ids_and_criterion_rejected(self):
        state = make_state(self.temp.name, ScenarioLLM())
        for plan in (task("retrieve_evidence", section_id="S1", source_ids=["P999"], query="cost"),
                     task("retrieve_evidence", section_id="missing", source_ids=["P001"], query="cost"),
                     {**task("human_review"), "success_criterion": "invented"}):
            with self.assertRaises(ValueError):
                validate_task(plan, state, ["retrieve_evidence", "human_review"])

    def test_invalid_planner_stops_visibly_and_budget_is_bounded(self):
        state = make_state(self.temp.name, MockLLM())
        plan_next_task(state)
        self.assertEqual(state.active_task["action"], "human_review")
        self.assertEqual(len(state.active_task["validation_errors"]), 2)
        state.config.max_agent_tasks = 1
        self.assertEqual(legal_actions(state), ["human_review"])

    def test_heuristic_audit_cannot_finalize_and_unknown_citation_is_not_hidden(self):
        state = make_state(self.temp.name, MockLLM())
        state.audits = audit_draft(state.draft, state.evidence_rows, MockLLM())
        state.draft_audited = state.coverage_checked = True
        self.assertNotIn("finalize", legal_actions(state))
        checks = audit_draft("A benchmark uses a fixed hardware configuration[P001,P999].", state.evidence_rows, ScenarioLLM())
        self.assertEqual(checks[0].verdict, "unknown_citation")

    def test_scoped_audit_keeps_unrelated_claim_ids(self):
        state = make_state(self.temp.name, ScenarioLLM())
        all_audits = audit_draft(state.draft, state.evidence_rows, state.llm)
        scoped = audit_draft(state.draft, state.evidence_rows, state.llm, section_filter="Evaluation setup")
        self.assertEqual(scoped[0].claim_id, all_audits[1].claim_id)

    def test_dirty_section_and_empty_section_block_completion(self):
        state = make_state(self.temp.name, ScenarioLLM())
        state.draft = state.draft.replace(BROAD, NARROW)
        state.audits = audit_draft(state.draft, state.evidence_rows, state.llm)
        state.draft_audited = state.coverage_checked = True
        state.dirty_sections = ["S1"]
        self.assertEqual(legal_actions(state), ["audit_section", "human_review"])
        with self.assertRaises(ValueError):
            validate_task(task("audit_section", section_id="S2"), state, legal_actions(state))
        state.dirty_sections.clear()
        state.evidence_dirty_sections = ["S1"]
        self.assertNotIn("finalize", legal_actions(state))
        state.evidence_dirty_sections.clear()
        state.audits = [a for a in state.audits if a.section != "Deployment costs"]
        self.assertNotIn("finalize", legal_actions(state))

    def test_stale_claim_and_repeated_task_rejected(self):
        state = make_state(self.temp.name, ScenarioLLM())
        with self.assertRaises(ValueError):
            validate_task(task("revise_claim", section_id="S1", claim_id="stale", evidence_ids=["E000001"],
                               instruction="Narrow"), state, ["revise_claim"])
        proposal = task("retrieve_evidence", section_id="S1", source_ids=["P001"], query="cost")
        validated = validate_task(proposal, state, ["retrieve_evidence"])
        state.task_history = [dict(validated), dict(validated)]
        with self.assertRaises(ValueError):
            validate_task(proposal, state, ["retrieve_evidence"])

    def test_planner_repairs_invalid_parameters_once(self):
        responses = iter(['{"action":"unknown"}', json.dumps(task("human_review"))])
        state = make_state(self.temp.name, ResponseLLM(lambda *_: next(responses)))
        plan_next_task(state)
        self.assertEqual(state.active_task["planner_status"], "llm_repaired")
        self.assertEqual(len(state.active_task["planning_attempts"]), 2)
        self.assertTrue(state.active_task["planning_attempts"][1]["accepted"])

    def test_langgraph_executes_the_same_task_tools(self):
        if not langgraph_available():
            self.skipTest("LangGraph is not installed")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langgraph.graph import END, StateGraph
            graph = StateGraph(GraphRuntimeState)
            for node in REVIEW_STEPS:
                graph.add_node(node, make_langgraph_node(node))
            paths = {node: node for node in REVIEW_STEPS}
            paths["finalize"] = END
            for node in REVIEW_STEPS:
                graph.add_conditional_edges(node, route_langgraph_state, paths)
            graph.set_entry_point("plan_tasks")
            state = make_state(self.temp.name, ScenarioLLM())
            result = graph.compile().invoke({"runtime": state, "next_node": "plan_tasks"}, {"recursion_limit": 60})
        self.assertEqual(result["next_node"], "finalize")
        self.assertEqual(state.task_history[-1]["action"], "finalize")
        self.assertIn(NARROW, state.draft)

    def test_replanning_loop_retrieves_repairs_and_reaudits(self):
        state = make_state(self.temp.name, ScenarioLLM())
        original_tail = state.draft.split("## Evaluation setup", 1)[1]
        next_node = "plan_tasks"
        for _ in range(60):
            if next_node == "finalize":
                break
            run_node(state, next_node)
            next_node = route_after_node(state, next_node)
        self.assertEqual(next_node, "finalize")
        actions = [t["action"] for t in state.task_history]
        self.assertEqual(actions, ["audit_citations", "retrieve_evidence", "revise_claim",
                                   "audit_section", "coverage_critic", "finalize"])
        self.assertIn(NARROW, state.draft)
        self.assertNotIn(BROAD, state.draft)
        self.assertEqual(state.draft.split("## Evaluation setup", 1)[1], original_tail)
        self.assertTrue(all(a.verdict == "supports" for a in state.audits))
        self.assertTrue(all(r["status"] == "ok" for r in state.coverage_critique_rows if "status" in r))
        self.assertFalse(state.dirty_sections)
        paths = persist_outputs(state)
        self.assertTrue(Path(paths["task_report"]).is_file())
        self.assertTrue(Path(paths["evidence_state_sqlite"]).is_file())
        saved = json.loads(Path(paths["task_history_json"]).read_text(encoding="utf-8"))
        self.assertTrue(saved[-1]["outcome"]["criterion_met"])
        import sqlite3
        with sqlite3.connect(state.store.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM task_events").fetchone()[0], 12)

    def test_existing_evidence_can_skip_retrieval(self):
        state = make_state(self.temp.name, ScenarioLLM(retrieve=False))
        node = "plan_tasks"
        for _ in range(40):
            if node == "finalize":
                break
            run_node(state, node)
            node = route_after_node(state, node)
        actions = [t["action"] for t in state.task_history]
        self.assertNotIn("retrieve_evidence", actions)
        self.assertIn("revise_claim", actions)
        self.assertEqual(actions[-1], "finalize")

    def test_failed_source_read_returns_observation_without_new_evidence(self):
        state = make_state(self.temp.name, ScenarioLLM())
        state.documents[0].ref.document_path = str(Path(self.temp.name) / "missing.pdf")
        state.active_task = task("retrieve_evidence", section_id="S1", source_ids=["P001"], query="cost")
        state.active_task["task_id"] = "T0001"
        run_node(state, "retrieve_evidence")
        self.assertEqual(state.active_task["status"], "failed")
        self.assertEqual(len(state.evidence_rows), 1)
        self.assertIn("error", state.active_task["outcome"])

    def test_source_card_rejects_fabricated_quotes_and_derives_page(self):
        state = make_state(self.temp.name, ScenarioLLM())
        source_window = {"source_window_id": "W1", "page_hint": "2", "text": QUOTE}
        claims = [
            {"claim": "Cost was not measured", "evidence_quote": QUOTE, "page_hint": "999", "source_window_id": "W1"},
            {"claim": "Costs fell", "evidence_quote": "Deployment costs fell by 40 percent.", "source_window_id": "W1"},
        ]
        with patch("source_grounded_review.steps.source_reader.select_source_windows", return_value=[source_window]):
            card = model_source_card(state.documents[0], state.sections,
                                     ResponseLLM(json.dumps({"evidence_claims": claims})))
        self.assertEqual(len(card.evidence_claims), 1)
        self.assertEqual(card.evidence_claims[0]["page_hint"], "2")

    def test_binding_keeps_all_cited_sources(self):
        rows = [Evidence(f"E{i}", f"P00{i}", "Benchmark", "", "", 1.0,
                         "The evaluation used a fixed hardware configuration throughout the benchmark.", "")
                for i in range(1, 5)]
        audits = audit_draft("The evaluation used a fixed hardware configuration throughout the benchmark[P001,P002,P003,P004].",
                             rows, ScenarioLLM())
        bindings = build_citation_bindings(audits, rows)
        self.assertEqual({b["citation_id"] for b in bindings if b["binding_status"] == "bound"},
                         {"P001", "P002", "P003", "P004"})


if __name__ == "__main__":
    unittest.main()
