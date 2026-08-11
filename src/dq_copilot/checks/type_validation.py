from collections.abc import Mapping

import pandas as pd

from dq_copilot.models import (
    ColumnTypeValidationResult,
    ExpectedDataType,
    InvalidTypeValueExample,
    TypeValidationCheckResult,
)


SUPPORTED_EXPECTED_TYPES = {
    "integer",
    "float",
    "numeric",
    "date",
    "string",
    "boolean",
}


def classify_type_validation_severity(
    invalid_percentage: float,
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
) -> str:
    """
    Classify type validation severity based on invalid value percentage.

    Rules:
    - 0% invalid      -> ok
    - <= warning      -> low
    - <= high         -> medium
    - > high          -> high
    """
    if invalid_percentage == 0:
        return "ok"

    if invalid_percentage <= warning_threshold:
        return "low"

    if invalid_percentage <= high_threshold:
        return "medium"

    return "high"


def _validate_expected_schema(
    dataframe: pd.DataFrame,
    expected_schema: Mapping[str, str],
) -> None:
    """
    Validate that expected schema columns and expected types are supported.
    """
    missing_columns = [
        column for column in expected_schema if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Expected schema columns not found: {missing_columns}")

    unsupported_types = sorted(
        {
            expected_type
            for expected_type in expected_schema.values()
            if expected_type not in SUPPORTED_EXPECTED_TYPES
        }
    )

    if unsupported_types:
        raise ValueError(f"Unsupported expected types: {unsupported_types}")


def _get_non_null_values(series: pd.Series) -> pd.Series:
    """
    Return only non-null values.

    Missing values are handled by the missing-value checker.
    Type validation focuses on values that are actually present.
    """
    return series[series.notna()]


def _build_invalid_mask(values: pd.Series, expected_type: str) -> pd.Series:
    """
    Return a boolean mask where True means the value is invalid.
    """
    if values.empty:
        return pd.Series([], dtype=bool, index=values.index)

    if expected_type == "string":
        return pd.Series(False, index=values.index)

    if expected_type in {"float", "numeric"}:
        converted = pd.to_numeric(values, errors="coerce")
        return converted.isna()

    if expected_type == "integer":
        converted = pd.to_numeric(values, errors="coerce")
        numeric_is_invalid = converted.isna()

        integer_is_invalid = converted % 1 != 0

        return numeric_is_invalid | integer_is_invalid

    if expected_type == "date":
        converted = pd.to_datetime(values, errors="coerce")
        return converted.isna()

    if expected_type == "boolean":
        accepted_values = {
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "y",
            "n",
        }

        normalized_values = values.astype(str).str.strip().str.lower()
        return ~normalized_values.isin(accepted_values)

    raise ValueError(f"Unsupported expected type: {expected_type}")


def _build_invalid_examples(
    invalid_values: pd.Series,
    expected_type: ExpectedDataType,
    max_examples: int,
) -> list[InvalidTypeValueExample]:
    """
    Build display-safe examples of invalid values.
    """
    examples: list[InvalidTypeValueExample] = []

    for row_index, value in invalid_values.head(max_examples).items():
        examples.append(
            InvalidTypeValueExample(
                row_index=str(row_index),
                value=str(value),
                expected_type=expected_type,
            )
        )

    return examples


def check_type_validation(
    dataframe: pd.DataFrame,
    expected_schema: Mapping[str, ExpectedDataType],
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
    max_examples: int = 10,
) -> TypeValidationCheckResult:
    """
    Validate DataFrame values against an expected schema.

    Parameters
    ----------
    dataframe:
        Dataset to analyze.

    expected_schema:
        Mapping between column names and expected data types.

        Example:
        {
            "customer_id": "integer",
            "age": "integer",
            "signup_date": "date"
        }

    warning_threshold:
        Percentage threshold above which invalid values become medium severity.

    high_threshold:
        Percentage threshold above which invalid values become high severity.

    max_examples:
        Maximum number of invalid values to include as examples.

    Returns
    -------
    TypeValidationCheckResult
        Structured type validation result.
    """
    if warning_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be positive numbers.")

    if warning_threshold > high_threshold:
        raise ValueError("warning_threshold cannot be greater than high_threshold.")

    if max_examples < 0:
        raise ValueError("max_examples must be a positive number.")

    _validate_expected_schema(dataframe, expected_schema)

    column_results: list[ColumnTypeValidationResult] = []

    for column_name, expected_type in expected_schema.items():
        series = dataframe[column_name]
        non_null_values = _get_non_null_values(series)

        invalid_mask = _build_invalid_mask(
            values=non_null_values,
            expected_type=expected_type,
        )

        invalid_values = non_null_values[invalid_mask]
        invalid_count = int(invalid_mask.sum())
        non_null_count = int(len(non_null_values))

        if non_null_count == 0:
            invalid_percentage = 0.0
        else:
            invalid_percentage = round((invalid_count / non_null_count) * 100, 2)

        severity = classify_type_validation_severity(
            invalid_percentage=invalid_percentage,
            warning_threshold=warning_threshold,
            high_threshold=high_threshold,
        )

        column_results.append(
            ColumnTypeValidationResult(
                column_name=str(column_name),
                expected_type=expected_type,
                pandas_dtype=str(series.dtype),
                non_null_count=non_null_count,
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                severity=severity,
                invalid_examples=_build_invalid_examples(
                    invalid_values=invalid_values,
                    expected_type=expected_type,
                    max_examples=max_examples,
                ),
            )
        )

    total_invalid_values = sum(result.invalid_count for result in column_results)
    columns_with_invalid_values = sum(
        1 for result in column_results if result.invalid_count > 0
    )

    has_high_severity = any(result.severity == "high" for result in column_results)

    if total_invalid_values == 0:
        status = "passed"
    elif has_high_severity:
        status = "failed"
    else:
        status = "warning"

    return TypeValidationCheckResult(
        status=status,
        columns_checked=len(expected_schema),
        columns_with_invalid_values=columns_with_invalid_values,
        total_invalid_values=total_invalid_values,
        results=column_results,
    )