from __future__ import annotations

import re
from collections.abc import Sequence

import duckdb
import pandas as pd

from dq_copilot.models import (
    SQLAnalysisQuery,
    SQLAnalysisQueryResult,
    SQLAnalysisRunResult,
)


FORBIDDEN_SQL_KEYWORDS = {
    "alter",
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "export",
    "import",
    "insert",
    "install",
    "load",
    "pragma",
    "set",
    "update",
}


def _normalize_sql(sql: str) -> str:
    """
    Normalize SQL text for basic validation.
    """
    return sql.strip().rstrip(";")


def _extract_sql_words(sql: str) -> set[str]:
    """
    Extract lowercase words from a SQL query.
    """
    return set(re.findall(r"\b[a-zA-Z_]+\b", sql.lower()))


def validate_read_only_sql(sql: str) -> None:
    """
    Validate that a SQL query is read-only.

    This is intentionally conservative because future LLM-generated SQL
    should not be allowed to execute destructive or external commands.
    """
    normalized_sql = _normalize_sql(sql)
    lowered_sql = normalized_sql.lower()

    if not lowered_sql:
        raise ValueError("SQL query cannot be empty.")

    if not lowered_sql.startswith(("select", "with")):
        raise ValueError("Only SELECT or WITH queries are allowed.")

    words = _extract_sql_words(lowered_sql)
    forbidden_words = sorted(words.intersection(FORBIDDEN_SQL_KEYWORDS))

    if forbidden_words:
        raise ValueError(f"Forbidden SQL keywords found: {forbidden_words}")


def _dataframe_rows_to_strings(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """
    Convert a DataFrame result into display-safe string dictionaries.
    """
    rows: list[dict[str, str]] = []

    for record in dataframe.to_dict(orient="records"):
        rows.append(
            {
                str(column): "" if pd.isna(value) else str(value)
                for column, value in record.items()
            }
        )

    return rows


def run_duckdb_query(
    dataframe: pd.DataFrame,
    query: SQLAnalysisQuery,
    table_name: str = "dataset",
    max_rows: int = 50,
) -> SQLAnalysisQueryResult:
    """
    Run a single read-only DuckDB SQL query against a Pandas DataFrame.

    Parameters
    ----------
    dataframe:
        Dataset to query.

    query:
        SQL query configuration.

    table_name:
        Name exposed to DuckDB SQL.

    max_rows:
        Maximum number of result rows kept for display.

    Returns
    -------
    SQLAnalysisQueryResult
        Structured SQL query result.
    """
    if max_rows < 0:
        raise ValueError("max_rows must be a positive number.")

    try:
        validate_read_only_sql(query.sql)

        connection = duckdb.connect(database=":memory:")
        connection.register(table_name, dataframe)

        result_dataframe = connection.execute(query.sql).df()
        result_preview = result_dataframe.head(max_rows)

        return SQLAnalysisQueryResult(
            query_name=query.query_name,
            sql=query.sql,
            status="passed",
            row_count=len(result_dataframe),
            columns=[str(column) for column in result_dataframe.columns],
            rows=_dataframe_rows_to_strings(result_preview),
        )

    except Exception as error:
        return SQLAnalysisQueryResult(
            query_name=query.query_name,
            sql=query.sql,
            status="failed",
            error_message=str(error),
        )

    finally:
        try:
            connection.close()
        except Exception:
            pass


def run_duckdb_analysis(
    dataframe: pd.DataFrame,
    queries: Sequence[SQLAnalysisQuery],
    table_name: str = "dataset",
    max_rows: int = 50,
) -> SQLAnalysisRunResult:
    """
    Run multiple DuckDB SQL analysis queries against a Pandas DataFrame.
    """
    query_results = [
        run_duckdb_query(
            dataframe=dataframe,
            query=query,
            table_name=table_name,
            max_rows=max_rows,
        )
        for query in queries
    ]

    queries_failed = sum(1 for result in query_results if result.status == "failed")

    if not query_results:
        status = "passed"
    elif queries_failed == 0:
        status = "passed"
    elif queries_failed == len(query_results):
        status = "failed"
    else:
        status = "warning"

    return SQLAnalysisRunResult(
        status=status,
        queries_executed=len(query_results),
        queries_failed=queries_failed,
        results=query_results,
    )