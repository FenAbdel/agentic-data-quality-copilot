import pandas as pd

from dq_copilot.models import ColumnMissingValueResult, MissingValuesCheckResult


def classify_missing_value_severity(
    null_percentage: float,
    warning_threshold: float = 5.0,
    high_threshold: float = 20.0,
) -> str:
    """
    Classify missing-value severity based on the percentage of missing values.

    Rules:
    - 0% missing      -> ok
    - <= warning      -> low
    - <= high         -> medium
    - > high          -> high
    """
    if null_percentage == 0:
        return "ok"

    if null_percentage <= warning_threshold:
        return "low"

    if null_percentage <= high_threshold:
        return "medium"

    return "high"


def check_missing_values(
    dataframe: pd.DataFrame,
    warning_threshold: float = 5.0,
    high_threshold: float = 20.0,
) -> MissingValuesCheckResult:
    """
    Analyze missing values in a DataFrame.

    Parameters
    ----------
    dataframe:
        Dataset to analyze.

    warning_threshold:
        Percentage threshold above which missing values become medium severity.

    high_threshold:
        Percentage threshold above which missing values become high severity.

    Returns
    -------
    MissingValuesCheckResult
        Structured missing-value analysis result.
    """
    if warning_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be positive numbers.")

    if warning_threshold > high_threshold:
        raise ValueError("warning_threshold cannot be greater than high_threshold.")

    row_count = len(dataframe)
    column_results: list[ColumnMissingValueResult] = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]

        null_count = int(series.isna().sum())

        if row_count == 0:
            null_percentage = 0.0
        else:
            null_percentage = round((null_count / row_count) * 100, 2)

        severity = classify_missing_value_severity(
            null_percentage=null_percentage,
            warning_threshold=warning_threshold,
            high_threshold=high_threshold,
        )

        column_results.append(
            ColumnMissingValueResult(
                column_name=str(column_name),
                null_count=null_count,
                null_percentage=null_percentage,
                severity=severity,
            )
        )

    total_missing_values = sum(result.null_count for result in column_results)
    columns_with_missing_values = sum(
        1 for result in column_results if result.null_count > 0
    )

    has_high_severity = any(result.severity == "high" for result in column_results)

    if total_missing_values == 0:
        status = "passed"
    elif has_high_severity:
        status = "failed"
    else:
        status = "warning"

    return MissingValuesCheckResult(
        total_missing_values=total_missing_values,
        columns_checked=len(dataframe.columns),
        columns_with_missing_values=columns_with_missing_values,
        results=column_results,
        status=status,
    )