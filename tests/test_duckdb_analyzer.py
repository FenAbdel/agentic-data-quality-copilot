import pandas as pd
import pytest

from dq_copilot.models import SQLAnalysisQuery
from dq_copilot.sql.duckdb_analyzer import (
    run_duckdb_analysis,
    run_duckdb_query,
    validate_read_only_sql,
)


def test_duckdb_query_runs_select_against_dataframe():
    dataframe = pd.DataFrame(
        {
            "country": ["France", "France", "Morocco"],
            "amount": [100, 200, 50],
        }
    )

    query = SQLAnalysisQuery(
        query_name="rows_by_country",
        sql="""
        SELECT country, COUNT(*) AS row_count, SUM(amount) AS total_amount
        FROM dataset
        GROUP BY country
        ORDER BY row_count DESC
        """,
    )

    result = run_duckdb_query(dataframe, query)

    assert result.status == "passed"
    assert result.query_name == "rows_by_country"
    assert result.row_count == 2
    assert result.columns == ["country", "row_count", "total_amount"]
    assert result.rows[0]["country"] == "France"
    assert result.rows[0]["row_count"] == "2"


def test_duckdb_analysis_runs_multiple_queries():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "country": ["France", "Morocco", "France"],
        }
    )

    queries = [
        SQLAnalysisQuery(
            query_name="total_rows",
            sql="SELECT COUNT(*) AS row_count FROM dataset",
        ),
        SQLAnalysisQuery(
            query_name="distinct_countries",
            sql="SELECT COUNT(DISTINCT country) AS country_count FROM dataset",
        ),
    ]

    result = run_duckdb_analysis(dataframe, queries)

    assert result.status == "passed"
    assert result.queries_executed == 2
    assert result.queries_failed == 0
    assert result.results[0].rows[0]["row_count"] == "3"
    assert result.results[1].rows[0]["country_count"] == "2"


def test_duckdb_query_returns_failed_result_for_invalid_sql():
    dataframe = pd.DataFrame({"a": [1, 2, 3]})

    query = SQLAnalysisQuery(
        query_name="invalid_query",
        sql="SELECT missing_column FROM dataset",
    )

    result = run_duckdb_query(dataframe, query)

    assert result.status == "failed"
    assert result.error_message is not None


def test_validate_read_only_sql_accepts_select_and_with():
    validate_read_only_sql("SELECT COUNT(*) FROM dataset")
    validate_read_only_sql(
        """
        WITH country_counts AS (
            SELECT country, COUNT(*) AS row_count
            FROM dataset
            GROUP BY country
        )
        SELECT * FROM country_counts
        """
    )


def test_validate_read_only_sql_rejects_write_or_admin_statements():
    with pytest.raises(ValueError):
        validate_read_only_sql("DROP TABLE dataset")

    with pytest.raises(ValueError):
        validate_read_only_sql("DELETE FROM dataset")

    with pytest.raises(ValueError):
        validate_read_only_sql("CREATE TABLE x AS SELECT * FROM dataset")

    with pytest.raises(ValueError):
        validate_read_only_sql("COPY dataset TO 'output.csv'")


def test_validate_read_only_sql_rejects_empty_query():
    with pytest.raises(ValueError):
        validate_read_only_sql("")