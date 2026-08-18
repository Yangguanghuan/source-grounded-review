# Source-Grounded Review

A small tool for drafting source-grounded reviews and research reports from local files. It reads the files you provide, summarizes what each source can support, maps sources to an outline, drafts the text, and exports citation checks so the final claims can be traced back to source snippets.

It is useful for literature reviews, technical research, market or policy reports, meeting-note synthesis, and other writing tasks where citations need to point back to the original material.

## What It Does

- Reads PDF, TXT, Markdown, JSON, JSONL, CSV, TSV, HTML, and other text-based files.
- Builds a structured source card for each file, including topic tags, method tags, useful sections, key findings, and limitations.
- Extracts citable evidence snippets with source paths, page hints, and related outline sections.
- Selects representative evidence for each outline section while reducing repeated use of the same source.
- Drafts a review or research report from the selected evidence.
- Checks whether cited statements are supported by the corresponding evidence.
- Can use an LLM planner to choose the next review step from guarded candidate actions.
- Exports citation checks, citation bindings, coverage reports, and manual review notes.

## Project Layout

```text
.
├── src/source_grounded_review/
│   ├── steps/            # Source reading, outline matching, evidence selection, writing, and checks
│   ├── core/             # Data models, model client, SQLite records, and utility functions
│   ├── io/               # Source loading, reference CSV handling, and outline parsing
│   ├── orchestration/    # One-pass pipeline and review loop
│   ├── reporting/        # Run summaries
│   └── cli.py            # Command-line entry point
├── examples/             # Small runnable example
├── scripts/              # Filename normalization helpers
├── data/inputs/.gitkeep  # Placeholder for local input files
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template:

```powershell
copy .env.example .env
```

Example `.env`:

```env
LLM_PROVIDER=mock
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
```

## Quick Start

Run the included mixed-format example:

```powershell
$env:PYTHONPATH="src"

python -m source_grounded_review.cli `
  --topic "Digital governance for community elderly-care services: practice, risk, and evaluation" `
  --input-dir examples\inputs `
  --outline examples\outline.md `
  --output-dir outputs\demo `
  --provider mock
```

If the package is installed, you can also use the script entry point:

```powershell
evidence-review --topic "Your research topic" --input-dir data\inputs --output-dir outputs\run --provider mock
```

## DeepSeek

Configure `.env` first:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-chat
```

Then run:

```powershell
$env:PYTHONPATH="src"

python -m source_grounded_review.cli `
  --topic "Your research topic" `
  --input-dir data\inputs `
  --outline examples\outline.md `
  --output-dir outputs\deepseek_run `
  --provider deepseek `
  --model-select-evidence `
  --planner llm
```

`--references` is optional. If it is not provided, all readable files under `--input-dir` are treated as pre-selected sources and assigned IDs such as `P001`, `P002`, and so on.

## Planner Mode

The review loop can route steps in two ways:

```powershell
--planner rules
```

Uses deterministic routing. This is the default mode and is useful for repeatable local runs.

```powershell
--planner llm
```

Uses the configured model as a planning controller. After each step, the system summarizes the current run state, shows the model the legal candidate actions, and asks it to choose the next step. The selected action is accepted only if it belongs to the candidate set; otherwise the guarded default route is used.

Planner decisions are written to `trace/step_decisions.csv`, including the selected action, candidate actions, planner status, and a shortened raw model response.

## Planning Controller

The planning controller is the part of the project that decides what should happen after each step. It is built around a shared `EvidenceState`, which stores the current corpus, source cards, evidence rows, outline assignments, section evidence packs, draft text, citation-audit results, coverage warnings, revision rounds, and manual-review items.

When `--planner llm` is enabled, the controller sends the model a compact planning prompt with:

- the step that just finished;
- a summary of the current `EvidenceState`;
- the current citation and coverage risk state;
- the legal candidate actions for the next step;
- the most recent route decisions.

The model returns a JSON decision:

```json
{
  "next_action": "rewrite_draft",
  "reason": "Citation audit found high-risk claims and one rewrite round is still available."
}
```

The system accepts the action only if it is one of the legal candidates. If the model returns an invalid action or malformed JSON, the guarded default route is used and the fallback is recorded.

Example:

```text
After: build_section_packs
State: 3 sections have weak evidence coverage, evidence expansion is still available
Candidate actions: expand_evidence, write_draft, human_review
Likely planner decision: expand_evidence
Next: broaden evidence budgets, then return to outline_mapper
```

Another example:

```text
After: audit_citations
State: flagged_claim_ratio = 0.52, misaligned_claims = 5, rewrite rounds remain
Candidate actions: rewrite_draft, coverage_critic, human_review
Likely planner decision: rewrite_draft
Next: rewrite the draft with audit feedback, then audit citations again
```

## Basic Flow

1. Load source files provided by the user.
2. Build a source card for each file.
3. Extract citable evidence snippets.
4. Match sources and evidence to outline sections.
5. Select representative evidence for each section.
6. Draft the review or research report.
7. Check whether citations match the supporting evidence.
8. Export coverage reports and manual review notes.

The default command runs the review loop, which can broaden evidence selection, rewrite the draft, or leave manual review notes when checks fail. For a faster one-pass run, add:

```powershell
--linear
```

## Step Inputs and Outputs

| Step | Main input | Main output |
|---|---|---|
| `load_corpus` | Topic, input directory, optional reference CSV, optional outline file | `documents`, `sections` |
| `build_source_cards` | Loaded documents and outline sections | `source_cards` |
| `build_evidence_matrix` | Documents, source cards, topic, outline sections | `evidence_rows` |
| `outline_mapper` | Source cards, evidence rows, outline sections | `assignments`, `outline_map` |
| `select_evidence` | Assignments, evidence rows, source cards, section limits | `selected_evidence_by_section`, `selection_rows` |
| `build_section_packs` | Sections, assignments, selected evidence | `section_packs` |
| `expand_evidence` | Current section-coverage state and selection budgets | Larger evidence-selection budgets |
| `write_draft` | Topic, outline sections, section evidence packs | `draft` |
| `audit_citations` | Draft and evidence matrix | `citation_audit_results`, `citation_bindings` |
| `rewrite_draft` | Draft, section packs, citation-audit feedback | Revised `draft` |
| `coverage_critic` | Draft, documents, assignments, section packs | `coverage_report`, source-usage rows |
| `human_review` | Current risk state | `manual_review_notes.md` |
| `finalize` | Full `EvidenceState` | Output folder with draft, tables, trace, and reports |

## Main Outputs

```text
draft/review_draft.md                   # Generated draft
audit/citation_audit_results.csv        # Citation check table
bindings/citation_bindings.csv          # Sentence -> citation -> evidence binding
evidence/evidence_matrix.csv            # Evidence snippet table
corpus/source_cards.json                # Source cards
packs/section_evidence_packs.json       # Evidence pack used by each section
coverage/coverage_report.md             # Section coverage report
manual_review/manual_review_notes.md    # Items that need manual review
trace/step_decisions.csv                # Why each step continued, looped back, or finished
report/run_summary.md                   # Run summary
```

## Common Options

- `--max-sources 5`: read only the first five sources for a small trial run.
- `--evidence-per-source 10`: keep up to ten evidence snippets per source.
- `--max-refs-per-section 6`: use up to six sources per outline section.
- `--max-evidence-per-section 10`: use up to ten evidence snippets per section.
- `--model-select-evidence`: use the configured model to help select final section evidence. Without this flag, rule-based ranking is used.
- `--planner llm`: use the configured model to choose the next review-loop step from guarded candidate actions.
- `--max-revision-rounds 2`: maximum rewrite rounds after citation checks fail.
- `--max-evidence-expansion-rounds 2`: maximum rounds for broadening evidence selection when section coverage is weak.

## Filename Normalization

If input PDF filenames are messy, copy them into a new directory with names like `P001_title.pdf`:

```powershell
$env:PYTHONPATH="src"

python scripts\normalize_pdf_filenames.py `
  --input-dir data\inputs `
  --out-dir data\normalized
```

If you already have a reference CSV, use it to normalize filenames by reference ID and title:

```powershell
$env:PYTHONPATH="src"

python scripts\normalize_source_filenames.py `
  --references references.csv `
  --input-dir data\inputs `
  --out-dir data\normalized
```
