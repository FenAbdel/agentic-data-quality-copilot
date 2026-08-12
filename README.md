# Agentic Data Quality Copilot

An agentic data quality assistant that profiles datasets, plans data quality checks, executes deterministic tools, verifies its own results, and generates clear reports for data engineering and BI use cases.

## Goal

This project is designed to demonstrate a realistic agentic AI workflow for data quality:

1. Inspect a dataset
2. Understand its schema
3. Plan relevant data quality checks
4. Execute checks with deterministic tools
5. Verify results
6. Generate a clear report
7. Keep an action log of what the agent did

## Current status

The full pipeline works end-to-end: profiling → planning → execution →
verification → reporting, driven from a Streamlit app.

- A **deterministic rule-based planner** picks which checks to run just by
  looking at the schema (this is the planner wired into the Streamlit app
  today).
- An **LLM planner** (OpenAI, via `.env` configuration) is implemented and
  tested — it proposes the same kind of plan by reasoning about the schema —
  but it is not yet selectable from the Streamlit UI. It always falls back to
  the deterministic planner if the LLM is unavailable or returns something
  invalid.
- A **verification layer** independently recomputes every result from the raw
  DataFrame and flags the run as `passed` / `warning` / `failed` if anything
  is inconsistent, so nothing reaches the report unchecked.

## MVP progress

- [x] CSV loading
- [x] Schema inspection
- [x] Missing value analysis
- [x] Duplicate detection
- [x] Type validation
- [x] Configurable business rules
- [x] DuckDB SQL checks (read-only, sandboxed)
- [x] BI-readiness scoring
- [x] Action logs
- [x] Markdown report generation
- [x] Streamlit demo (manual mode + Auto Copilot mode)
- [x] Deterministic rule-based planner
- [x] Result verification / self-check layer
- [x] LLM-based planner (implemented, tested, not yet wired into the UI)
- [ ] Let the Streamlit UI choose between the deterministic and LLM planner

## Architecture, in plain language

Think of it as **three roles working together**, all built so the "smart"
part (an LLM) is never the part doing arithmetic:

1. **The Planner decides *what* to check.**
   Today that's a deterministic, rule-based planner: it looks at the
   dataset's columns and types and decides things like "these look like ID
   columns, check them for duplicates" or "this column looks numeric, expect
   a number type." An LLM-based planner exists too — it does the same job by
   reasoning over the schema instead of using fixed rules — but it never sees
   the raw data, only column names/types/null percentages, and it can only
   *propose* a plan, never invent results.

2. **The Executor does the *math*.**
   Plain Python/pandas/DuckDB functions actually run each check (missing
   values, duplicates, type mismatches, business rules, SQL queries) against
   the real data and return a strict, typed result — never free text. This
   is the same code path regardless of which planner produced the plan.

3. **The Verifier checks the Executor's homework.**
   After the checks run, a separate deterministic layer independently
   recomputes totals straight from the DataFrame (e.g., "does the reported
   missing-value count actually match what's in the data?") and marks the
   whole run `passed`, `warning`, or `failed`. This exists so a bug or a
   bad LLM suggestion can't silently produce a wrong report.

On top of that: an **Observer** turns the verified results into a short list
of plain-English observations ("found 12 duplicate rows using scope X"), and
a **Reporter** turns everything into a downloadable Markdown report. A
Streamlit app ties it all together as the UI.

**Engineering principle:** the LLM should plan and explain, but Python,
Pandas, and DuckDB should calculate and verify the results.

## Architecture diagram

```
CSV file
   │
   ▼
Loader → pandas DataFrame
   │
   ▼
Schema Profiler  (columns, dtypes, null %, sample values)
   │
   ▼
Planner  ── picks which checks to run ──────────────────────┐
   │  (deterministic rules today; LLM planner available,     │
   │   not yet wired into the UI — always falls back to      │
   │   deterministic rules if it fails)                       │
   ▼                                                          │
Check Runner (src/dq_copilot/agent/check_runner.py)           │
   ├─ Missing values check                                    │
   ├─ Duplicate check                                          │
   ├─ Type validation check     (optional — needs expected schema)
   ├─ Business rules check      (optional — needs configured rules)
   └─ DuckDB SQL analysis       (optional — needs SQL queries, read-only)
   │
   ▼
BI-Readiness Scorer (combines every check into one 0-100 score)
   │
   ▼
Result Verifier  ── recomputes totals from the raw DataFrame and
   │                 marks the run passed / warning / failed
   ▼
Observer  ── turns verified results into plain-English observations
   │
   ▼
Markdown Report  +  Streamlit dashboard (two views of the same result)
```

Key ideas:

- **Every check returns a typed Pydantic model** (`src/dq_copilot/models.py`) —
  status, counts, severity — never free text. That makes results reliable for
  an LLM to read or write, instead of parsing prose.
- **The check runner acts like an executed plan.** `run_data_quality_checks()`
  runs each check in sequence and records an `action_log` entry per step —
  the audit trail the agent uses to explain what it did and why.
- **SQL is sandboxed.** The DuckDB layer (`src/dq_copilot/sql/duckdb_analyzer.py`)
  only allows `SELECT`/`WITH` queries and blocklists destructive keywords
  (`DROP`, `INSERT`, `PRAGMA`, etc.), since this is the part most likely to
  take LLM-generated SQL as input.
- **The LLM planner never computes metrics.** It only receives column
  names/dtypes/null percentages (`src/dq_copilot/agent/llm_planner.py`),
  proposes a `DataQualityRunConfig`, and its output is Pydantic-validated
  and column-checked before anything runs. If it fails or returns something
  invalid, the run automatically falls back to the deterministic planner.
- **Verification is independent of execution.** `result_verifier.py`
  recomputes each metric directly from the DataFrame rather than trusting
  the check that produced it, so a bug in one layer is likely to be caught
  by the other.
- **Streamlit is the demo/control surface** (`app/streamlit_app.py`) — upload
  a CSV (or use the built-in sample dataset), choose Auto Copilot or Manual
  mode, view tabbed results (plan, observations, schema, each check,
  BI-readiness, verification, action log, report), and download the
  Markdown report.

## Project layout

```
src/dq_copilot/
  loaders/          CSV loading
  profiling/        Schema profiling
  checks/           Missing values, duplicates, type validation, business rules
  sql/              Sandboxed DuckDB SQL analysis
  scoring/          BI-readiness scoring
  agent/
    deterministic_rules_planner.py   Rule-based planner
    llm_planner.py                   LLM-based planner (OpenAI)
    planner_factory.py               Chooses a planner implementation
    check_runner.py                  Executes the deterministic check pipeline
    result_verifier.py               Recomputes and verifies results
    observer.py                      Builds plain-English observations
    copilot.py                       Ties planning → execution → verification → report together
  reporting/        Markdown report generation
  models.py          Shared Pydantic models for every result type
app/streamlit_app.py Streamlit UI (manual mode + Auto Copilot mode)
tests/               pytest suite covering every layer above
```

## Running it

```bash
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

To try the LLM planner, copy `.env.example` to `.env` and set
`OPENAI_API_KEY` (it is not yet selectable from the UI, but it can be used
directly via `run_copilot_analysis(..., planner=LLMPlanner())` or
`create_planner("llm")`).

Run the test suite with:

```bash
pytest
```

## Engineering principle

The LLM should plan and explain, but Python, Pandas, and DuckDB should calculate and verify the results.
