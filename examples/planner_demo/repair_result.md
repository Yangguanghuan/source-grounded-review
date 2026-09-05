# Recorded Planner Repair

This is a small DeepSeek integration check using synthetic source text and a deliberately incorrect draft claim. It demonstrates task execution, not model accuracy on a research corpus.

The repair fixture is defined in `tests/test_task_planner.py`. For this recorded run, DeepSeek replaced the scripted test model. The fixture contains two simulated extracted pages: a fixed-hardware evaluation note and an explicit statement that deployment cost was not measured. Only the hardware statement is initially available in the evidence matrix.

## Before

> The optimized runtime reduces deployment costs by 40%[P001].

The source contains no such measurement. A second section describes the fixed hardware configuration and should remain unchanged.

## Observed Tasks

| Task | Action | Observed result |
|---|---|---|
| T0001 | `audit_citations` | Flagged the cost claim as misaligned with the selected evidence. |
| T0002 | `retrieve_evidence` | Chose P001 and queried deployment costs; extracted a new original-source quote. |
| T0003 | `revise_claim` | Selected the new evidence ID and replaced the unsupported quantitative claim. |
| T0004 | `audit_section` | Re-audited only the cost section; the revised claim received `supports`. |
| T0005 | `coverage_critic` | Confirmed coverage under this fixture's one-source, one-evidence minimum. |
| T0006 | `finalize` | Completed without manual-review items. |

New evidence:

> Deployment cost was not measured in this benchmark.

The new evidence record includes its source ID, simulated page number, exact offsets in the extracted text, source context, and a SHA-256 digest of that text. The quote is checked against the source window before being accepted.

## After

> Deployment costs were not measured in the benchmark[P001].

The evaluation-setup section remained unchanged. The revision task was recorded as pending semantic audit until T0004 completed.

The final result had two model-audited supported claims and no unresolved review items. This is one synthetic case, not a claim of general factual reliability. Earlier integration runs exposed a missed limitation quote and conflicting coverage thresholds; those prompted extraction-instruction and threshold-consistency fixes.

## Full Generation Example

The adjacent `inputs/` files and `outline.md` exercise generation from files rather than starting with a seeded draft. They are also synthetic.

```powershell
$env:PYTHONPATH="src"
python -m source_grounded_review.cli `
  --topic "Runtime performance and deployment cost" `
  --input-dir examples/planner_demo/inputs `
  --outline examples/planner_demo/outline.md `
  --output-dir outputs/planner_demo `
  --provider deepseek `
  --planner llm `
  --max-agent-tasks 10 `
  --min-refs-per-section 1 `
  --min-evidence-per-section 1
```

In the integration check, this path read two files and completed with five model-audited supported claims. Its planner chose initial drafting, full citation audit, coverage checking, then finalization. No targeted repair was necessary.
