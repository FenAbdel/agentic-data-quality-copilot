from pathlib import Path

import pandas as pd
import pytest

from dq_copilot.checks.missing_values import (
    check_missing_values,
    classify_missing_value_severity,
)
from dq_copilot.loaders.csv_loader import load_csv


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_missing_values_check_detects_missing_values():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = check_missing_values(dataframe)

    assert result.check_name == "missing_values"
    assert result.status == "warning"
    assert result.total_missing_values == 2
    assert result.columns_checked == 6
    assert result.columns_with_missing_values == 2


def test_missing_values_check_returns_column_level_results():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = check_missing_values(dataframe)

    results_by_column = {column.column_name: column for column in result.results}

    assert results_by_column["email"].null_count == 1
    assert results_by_column["email"].null_percentage == 20.0
    assert results_by_column["email"].severity == "medium"

    assert results_by_column["age"].null_count == 1
    assert results_by_column["age"].null_percentage == 20.0
    assert results_by_column["age"].severity == "medium"


def test_missing_values_check_passes_when_no_missing_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Amina", "Karim", "Lina"],
            "country": ["France", "Morocco", "Spain"],
        }
    )

    result = check_missing_values(dataframe)

    assert result.status == "passed"
    assert result.total_missing_values == 0
    assert result.columns_with_missing_values == 0

    for column_result in result.results:
        assert column_result.severity == "ok"


def test_missing_values_check_fails_when_high_severity_exists():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "email": ["a@example.com", None, None, None, None],
        }
    )

    result = check_missing_values(dataframe)

    assert result.status == "failed"
    assert result.total_missing_values == 4

    results_by_column = {column.column_name: column for column in result.results}

    assert results_by_column["email"].null_percentage == 80.0
    assert results_by_column["email"].severity == "high"


def test_missing_value_severity_classification():
    assert classify_missing_value_severity(0.0) == "ok"
    assert classify_missing_value_severity(3.0) == "low"
    assert classify_missing_value_severity(10.0) == "medium"
    assert classify_missing_value_severity(30.0) == "high"


def test_missing_values_check_rejects_invalid_thresholds():
    dataframe = pd.DataFrame({"a": [1, None, 3]})

    with pytest.raises(ValueError):
        check_missing_values(dataframe, warning_threshold=-1)

    with pytest.raises(ValueError):
        check_missing_values(dataframe, warning_threshold=30, high_threshold=10)