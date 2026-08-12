# Agentic Data Quality Copilot

An agentic data quality assistant that profiles datasets, plans data quality checks, executes deterministic tools, verifies results, and generates clear reports for data engineering and BI use cases.

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

The deterministic execution layer is complete and covered by tests. There is no
LLM wired in yet — checks are currently selected by the user (via the Streamlit
UI or a `DataQualityRunConfig`), not planned by an agent.

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
- [x] Streamlit demo
- [ ] LLM-driven planning of which checks to run

## Architecture

The project is a pipeline with a control room, not yet a full agent — it's the
deterministic scaffolding an LLM will eventually plan and drive.

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
Check Runner (src/dq_copilot/agent/check_runner.py)
   ├─ Missing values check
   ├─ Duplicate check
   ├─ Type validation check     (optional — needs an expected schema)
   ├─ Business rules check      (optional — needs configured rules)
   └─ DuckDB SQL analysis       (optional — needs SQL queries)
   │
   ▼
BI-Readiness Scorer (combines every check into one 0-100 score)
   │
   ▼
Markdown Report  +  Streamlit dashboard (two views of the same result)
```

Key ideas:

- **Every check returns a typed Pydantic model** (`src/dq_copilot/models.py`) —
  status, counts, severity — never free text. That makes results reliable for
  an LLM to read or write later, instead of parsing prose.
- **The check runner acts like an executed plan.** `run_data_quality_checks()`
  runs each check in sequence and records an `action_log` entry per step, which
  is the audit trail an LLM-driven agent will need to explain what it did and
  why. Today the "plan" is fixed/config-driven, not chosen by an LLM.
- **SQL is sandboxed.** The DuckDB layer (`src/dq_copilot/sql/duckdb_analyzer.py`)
  only allows `SELECT`/`WITH` queries and blocklists destructive keywords
  (`DROP`, `INSERT`, `PRAGMA`, etc.), since this is the part most likely to
  eventually take LLM-generated SQL as input.
- **Streamlit is the demo/control surface** (`app/streamlit_app.py`) — upload a
  CSV, configure checks, view tabbed results, download the Markdown report.

## Engineering principle

The LLM should plan and explain, but Python, Pandas, and DuckDB should calculate and verify the results.