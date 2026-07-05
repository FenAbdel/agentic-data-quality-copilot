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

Project foundation initialized.

## Planned MVP

- CSV loading
- Schema inspection
- Missing value analysis
- Duplicate detection
- Type validation
- DuckDB SQL checks
- Action logs
- Markdown report generation
- Streamlit demo

## Engineering principle

The LLM should plan and explain, but Python, Pandas, and DuckDB should calculate and verify the results.