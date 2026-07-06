import pandas as pd
import pytest

from dq_copilot.checks.duplicates import (
    check_duplicates,
    classify_duplicate_severity,
)


def test_duplicate_check_passes_when_no_duplicates_exist():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Amina", "Karim", "Lina"],
            "country": ["France", "Morocco", "Spain"],
        }
    )

    result = check_duplicates(dataframe)

    assert result.check_name == "duplicate_detection"
    assert result.status == "passed"
    assert result.scope == "full_row"
    assert result.row_count == 3
    assert result.duplicate_row_count == 0
    assert result.duplicate_percentage == 0.0
    assert result.severity == "ok"
    assert result.duplicate_group_count == 0


def test_duplicate_check_detects_full_row_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "name": ["Amina", "Amina", "Karim"],
            "country": ["France", "France", "Morocco"],
        }
    )

    result = check_duplicates(dataframe)

    assert result.status == "failed"
    assert result.scope == "full_row"
    assert result.duplicate_row_count == 2
    assert result.duplicate_percentage == 66.67
    assert result.severity == "high"
    assert result.duplicate_group_count == 1

    first_group = result.duplicate_groups[0]

    assert first_group.duplicate_key == {
        "customer_id": "1",
        "name": "Amina",
        "country": "France",
    }
    assert first_group.duplicate_count == 2


def test_duplicate_check_detects_key_column_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "name": ["Amina", "Amina Updated", "Karim"],
            "country": ["France", "France", "Morocco"],
        }
    )

    result = check_duplicates(dataframe, key_columns=["customer_id"])

    assert result.status == "failed"
    assert result.scope == "key_columns"
    assert result.key_columns == ["customer_id"]
    assert result.duplicate_row_count == 2
    assert result.duplicate_percentage == 66.67
    assert result.duplicate_group_count == 1

    first_group = result.duplicate_groups[0]

    assert first_group.duplicate_key == {"customer_id": "1"}
    assert first_group.duplicate_count == 2


def test_duplicate_check_supports_composite_keys():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2],
            "order_id": [100, 100, 101, 200],
            "amount": [50, 50, 70, 90],
        }
    )

    result = check_duplicates(
        dataframe,
        key_columns=["customer_id", "order_id"],
    )

    assert result.status == "failed"
    assert result.scope == "key_columns"
    assert result.key_columns == ["customer_id", "order_id"]
    assert result.duplicate_row_count == 2
    assert result.duplicate_group_count == 1

    first_group = result.duplicate_groups[0]

    assert first_group.duplicate_key == {
        "customer_id": "1",
        "order_id": "100",
    }
    assert first_group.duplicate_count == 2


def test_duplicate_check_limits_duplicate_group_examples():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "name": ["A", "A2", "B", "B2", "C", "C2"],
        }
    )

    result = check_duplicates(
        dataframe,
        key_columns=["customer_id"],
        max_groups=2,
    )

    assert result.duplicate_group_count == 3
    assert len(result.duplicate_groups) == 2


def test_duplicate_check_rejects_unknown_key_columns():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Amina", "Karim", "Lina"],
        }
    )

    with pytest.raises(ValueError):
        check_duplicates(dataframe, key_columns=["unknown_column"])


def test_duplicate_severity_classification():
    assert classify_duplicate_severity(0.0) == "ok"
    assert classify_duplicate_severity(0.5) == "low"
    assert classify_duplicate_severity(3.0) == "medium"
    assert classify_duplicate_severity(10.0) == "high"


def test_duplicate_check_rejects_invalid_thresholds():
    dataframe = pd.DataFrame({"a": [1, 1, 2]})

    with pytest.raises(ValueError):
        check_duplicates(dataframe, warning_threshold=-1)

    with pytest.raises(ValueError):
        check_duplicates(dataframe, warning_threshold=10, high_threshold=5)

    with pytest.raises(ValueError):
        check_duplicates(dataframe, max_groups=-1)