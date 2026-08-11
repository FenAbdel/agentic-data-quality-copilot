import pandas as pd
import pytest

from dq_copilot.checks.type_validation import (
    check_type_validation,
    classify_type_validation_severity,
)


def test_type_validation_passes_when_values_match_expected_schema():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["1", "2", "3"],
            "age": ["29", "34", "41"],
            "signup_date": ["2024-01-15", "2024-02-20", "2024-03-05"],
            "is_active": ["true", "false", "yes"],
        }
    )

    expected_schema = {
        "customer_id": "integer",
        "age": "integer",
        "signup_date": "date",
        "is_active": "boolean",
    }

    result = check_type_validation(dataframe, expected_schema)

    assert result.check_name == "type_validation"
    assert result.status == "passed"
    assert result.columns_checked == 4
    assert result.columns_with_invalid_values == 0
    assert result.total_invalid_values == 0


def test_type_validation_detects_invalid_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["1", "2", "ABC", "4"],
            "age": ["29", "unknown", "41", None],
            "signup_date": ["2024-01-15", "not-a-date", "2024-03-05", None],
            "is_active": ["true", "false", "maybe", None],
        }
    )

    expected_schema = {
        "customer_id": "integer",
        "age": "integer",
        "signup_date": "date",
        "is_active": "boolean",
    }

    result = check_type_validation(dataframe, expected_schema)

    assert result.status == "failed"
    assert result.columns_checked == 4
    assert result.columns_with_invalid_values == 4
    assert result.total_invalid_values == 4

    results_by_column = {column.column_name: column for column in result.results}

    assert results_by_column["customer_id"].invalid_count == 1
    assert results_by_column["customer_id"].invalid_percentage == 25.0
    assert results_by_column["customer_id"].severity == "high"

    assert results_by_column["age"].invalid_count == 1
    assert results_by_column["age"].invalid_percentage == 33.33

    assert results_by_column["signup_date"].invalid_count == 1
    assert results_by_column["signup_date"].invalid_percentage == 33.33

    assert results_by_column["is_active"].invalid_count == 1
    assert results_by_column["is_active"].invalid_percentage == 33.33


def test_type_validation_keeps_invalid_examples():
    dataframe = pd.DataFrame(
        {
            "age": ["29", "unknown", "41"],
        }
    )

    expected_schema = {
        "age": "integer",
    }

    result = check_type_validation(dataframe, expected_schema)

    age_result = result.results[0]

    assert age_result.invalid_count == 1
    assert len(age_result.invalid_examples) == 1
    assert age_result.invalid_examples[0].row_index == "1"
    assert age_result.invalid_examples[0].value == "unknown"
    assert age_result.invalid_examples[0].expected_type == "integer"


def test_type_validation_limits_invalid_examples():
    dataframe = pd.DataFrame(
        {
            "age": ["bad_1", "bad_2", "bad_3"],
        }
    )

    expected_schema = {
        "age": "integer",
    }

    result = check_type_validation(
        dataframe,
        expected_schema,
        max_examples=2,
    )

    age_result = result.results[0]

    assert age_result.invalid_count == 3
    assert len(age_result.invalid_examples) == 2


def test_type_validation_ignores_missing_values():
    dataframe = pd.DataFrame(
        {
            "age": ["29", None, "41"],
        }
    )

    expected_schema = {
        "age": "integer",
    }

    result = check_type_validation(dataframe, expected_schema)

    age_result = result.results[0]

    assert result.status == "passed"
    assert age_result.non_null_count == 2
    assert age_result.invalid_count == 0
    assert age_result.invalid_percentage == 0.0


def test_type_validation_accepts_integer_like_float_values():
    dataframe = pd.DataFrame(
        {
            "age": [29.0, 34.0, 41.0],
        }
    )

    expected_schema = {
        "age": "integer",
    }

    result = check_type_validation(dataframe, expected_schema)

    assert result.status == "passed"
    assert result.total_invalid_values == 0


def test_type_validation_rejects_decimal_values_for_integer_type():
    dataframe = pd.DataFrame(
        {
            "age": [29.5, 34.0, 41.0],
        }
    )

    expected_schema = {
        "age": "integer",
    }

    result = check_type_validation(dataframe, expected_schema)

    age_result = result.results[0]

    assert result.status == "failed"
    assert age_result.invalid_count == 1
    assert age_result.invalid_examples[0].value == "29.5"


def test_type_validation_rejects_unknown_columns():
    dataframe = pd.DataFrame(
        {
            "age": ["29", "34"],
        }
    )

    expected_schema = {
        "unknown_column": "integer",
    }

    with pytest.raises(ValueError):
        check_type_validation(dataframe, expected_schema)


def test_type_validation_rejects_unsupported_expected_types():
    dataframe = pd.DataFrame(
        {
            "age": ["29", "34"],
        }
    )

    expected_schema = {
        "age": "email",
    }

    with pytest.raises(ValueError):
        check_type_validation(dataframe, expected_schema)


def test_type_validation_severity_classification():
    assert classify_type_validation_severity(0.0) == "ok"
    assert classify_type_validation_severity(0.5) == "low"
    assert classify_type_validation_severity(3.0) == "medium"
    assert classify_type_validation_severity(10.0) == "high"


def test_type_validation_rejects_invalid_thresholds():
    dataframe = pd.DataFrame({"age": ["29", "unknown"]})
    expected_schema = {"age": "integer"}

    with pytest.raises(ValueError):
        check_type_validation(dataframe, expected_schema, warning_threshold=-1)

    with pytest.raises(ValueError):
        check_type_validation(
            dataframe,
            expected_schema,
            warning_threshold=10,
            high_threshold=5,
        )

    with pytest.raises(ValueError):
        check_type_validation(dataframe, expected_schema, max_examples=-1)