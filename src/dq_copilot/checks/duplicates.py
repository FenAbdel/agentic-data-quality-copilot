from collections.abc import Sequence

import pandas as pd

from dq_copilot.models import DuplicateCheckResult, DuplicateGroupResult


def classify_duplicate_severity(
    duplicate_percentage: float,
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
) -> str:
    """
    Classify duplicate severity based on the percentage of duplicated rows.

    Rules:
    - 0% duplicates      -> ok
    - <= warning         -> low
    - <= high            -> medium
    - > high             -> high
    """
    if duplicate_percentage == 0:
        return "ok"

    if duplicate_percentage <= warning_threshold:
        return "low"

    if duplicate_percentage <= high_threshold:
        return "medium"

    return "high"


def _validate_key_columns(dataframe: pd.DataFrame, key_columns: Sequence[str]) -> None:
    """
    Validate that all requested key columns exist in the DataFrame.
    """
    missing_columns = [
        column for column in key_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Key columns not found in dataset: {missing_columns}")


def _stringify_duplicate_key(row: pd.Series, key_columns: list[str]) -> dict[str, str]:
    """
    Convert duplicate key values to strings for safe display in reports/UI.
    """
    return {column: str(row[column]) for column in key_columns}


def _build_duplicate_groups(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    max_groups: int,
) -> list[DuplicateGroupResult]:
    """
    Build example duplicate groups.

    Each group represents a duplicated key and how many times it appears.
    """
    if dataframe.empty or not key_columns:
        return []

    grouped = (
        dataframe.groupby(key_columns, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
    )

    duplicated_groups = grouped[grouped["duplicate_count"] > 1]
    duplicated_groups = duplicated_groups.sort_values(
        by="duplicate_count",
        ascending=False,
    )

    results: list[DuplicateGroupResult] = []

    for _, row in duplicated_groups.head(max_groups).iterrows():
        results.append(
            DuplicateGroupResult(
                duplicate_key=_stringify_duplicate_key(row, key_columns),
                duplicate_count=int(row["duplicate_count"]),
            )
        )

    return results


def check_duplicates(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
    max_groups: int = 10,
) -> DuplicateCheckResult:
    """
    Detect duplicate rows in a DataFrame.

    If key_columns is provided, duplicates are checked using those columns.
    If key_columns is not provided, full-row duplicates are checked.

    Parameters
    ----------
    dataframe:
        Dataset to analyze.

    key_columns:
        Optional list of columns that should uniquely identify a row.

    warning_threshold:
        Percentage threshold above which duplicates become medium severity.

    high_threshold:
        Percentage threshold above which duplicates become high severity.

    max_groups:
        Maximum number of duplicate groups to include as examples.

    Returns
    -------
    DuplicateCheckResult
        Structured duplicate detection result.
    """
    if warning_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be positive numbers.")

    if warning_threshold > high_threshold:
        raise ValueError("warning_threshold cannot be greater than high_threshold.")

    if max_groups < 0:
        raise ValueError("max_groups must be a positive number.")

    if key_columns:
        selected_key_columns = list(key_columns)
        _validate_key_columns(dataframe, selected_key_columns)
        scope = "key_columns"
    else:
        selected_key_columns = list(dataframe.columns)
        scope = "full_row"

    row_count = len(dataframe)

    if row_count == 0 or not selected_key_columns:
        return DuplicateCheckResult(
            status="passed",
            scope=scope,
            key_columns=selected_key_columns,
            row_count=row_count,
            duplicate_row_count=0,
            duplicate_percentage=0.0,
            duplicate_group_count=0,
            severity="ok",
            duplicate_groups=[],
        )

    duplicate_mask = dataframe.duplicated(
        subset=selected_key_columns,
        keep=False,
    )

    duplicate_row_count = int(duplicate_mask.sum())
    duplicate_percentage = round((duplicate_row_count / row_count) * 100, 2)

    severity = classify_duplicate_severity(
        duplicate_percentage=duplicate_percentage,
        warning_threshold=warning_threshold,
        high_threshold=high_threshold,
    )

    duplicate_groups = _build_duplicate_groups(
        dataframe=dataframe,
        key_columns=selected_key_columns,
        max_groups=max_groups,
    )

    duplicate_group_count = len(
        dataframe[dataframe.duplicated(subset=selected_key_columns, keep=False)]
        .groupby(selected_key_columns, dropna=False)
        .size()
    )

    if duplicate_row_count == 0:
        status = "passed"
    elif severity == "high":
        status = "failed"
    else:
        status = "warning"

    return DuplicateCheckResult(
        status=status,
        scope=scope,
        key_columns=selected_key_columns,
        row_count=row_count,
        duplicate_row_count=duplicate_row_count,
        duplicate_percentage=duplicate_percentage,
        duplicate_group_count=duplicate_group_count,
        severity=severity,
        duplicate_groups=duplicate_groups,
    )