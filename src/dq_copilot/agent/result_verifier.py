from __future__ import annotations

import pandas as pd

from dq_copilot.models import (
    DataQualityRunResult,
    VerificationCheckResult,
    VerificationRunResult,
)
from dq_copilot.sql.duckdb_analyzer import validate_read_only_sql


def _passed(check_name: str, message: str) -> VerificationCheckResult:
    return VerificationCheckResult(
        check_name=check_name,
        status="passed",
        message=message,
    )


def _warning(check_name: str, message: str) -> VerificationCheckResult:
    return VerificationCheckResult(
        check_name=check_name,
        status="warning",
        message=message,
    )


def _failed(check_name: str, message: str) -> VerificationCheckResult:
    return VerificationCheckResult(
        check_name=check_name,
        status="failed",
        message=message,
    )


def _verify_schema_counts(
    dataframe: pd.DataFrame,
    result: DataQualityRunResult,
) -> VerificationCheckResult:
    actual_rows = len(dataframe)
    actual_columns = len(dataframe.columns)

    profile = result.schema_profile

    if profile.row_count != actual_rows or profile.column_count != actual_columns:
        return _failed(
            "schema_counts",
            (
                "Schema profile does not match the DataFrame shape. "
                f"Profile has {profile.row_count} rows and {profile.column_count} columns, "
                f"but DataFrame has {actual_rows} rows and {actual_columns} columns."
            ),
        )

    return _passed(
        "schema_counts",
        "Schema profile row and column counts match the DataFrame.",
    )


def _verify_missing_values(
    dataframe: pd.DataFrame,
    result: DataQualityRunResult,
) -> VerificationCheckResult:
    actual_missing = int(dataframe.isna().sum().sum())
    reported_missing = result.missing_values.total_missing_values

    column_level_missing = sum(
        column.null_count for column in result.missing_values.results
    )

    if reported_missing != column_level_missing:
        return _failed(
            "missing_values_consistency",
            (
                "Missing value result is internally inconsistent. "
                f"Reported total is {reported_missing}, but column-level sum is "
                f"{column_level_missing}."
            ),
        )

    if reported_missing != actual_missing:
        return _failed(
            "missing_values_recomputation",
            (
                "Missing value total does not match recomputation from DataFrame. "
                f"Reported {reported_missing}, recomputed {actual_missing}."
            ),
        )

    return _passed(
        "missing_values",
        "Missing value totals match column-level results and DataFrame recomputation.",
    )


def _verify_duplicates(
    dataframe: pd.DataFrame,
    result: DataQualityRunResult,
) -> VerificationCheckResult:
    duplicate_result = result.duplicates
    subset = duplicate_result.key_columns

    if duplicate_result.row_count != len(dataframe):
        return _failed(
            "duplicate_row_count",
            (
                "Duplicate check row count does not match DataFrame row count. "
                f"Reported {duplicate_result.row_count}, actual {len(dataframe)}."
            ),
        )

    if len(dataframe) == 0 or not subset:
        recomputed_duplicate_count = 0
    else:
        missing_columns = [column for column in subset if column not in dataframe.columns]

        if missing_columns:
            return _failed(
                "duplicate_key_columns",
                f"Duplicate key columns are missing from DataFrame: {missing_columns}",
            )

        recomputed_duplicate_count = int(
            dataframe.duplicated(subset=subset, keep=False).sum()
        )

    if duplicate_result.duplicate_row_count != recomputed_duplicate_count:
        return _failed(
            "duplicates_recomputation",
            (
                "Duplicate row count does not match recomputation from DataFrame. "
                f"Reported {duplicate_result.duplicate_row_count}, "
                f"recomputed {recomputed_duplicate_count}."
            ),
        )

    return _passed(
        "duplicates",
        "Duplicate row count matches DataFrame recomputation.",
    )


def _verify_type_validation(result: DataQualityRunResult) -> VerificationCheckResult:
    if result.type_validation is None:
        return _passed(
            "type_validation",
            "Type validation was not configured, so no type validation consistency check was required.",
        )

    total_from_columns = sum(
        column.invalid_count for column in result.type_validation.results
    )

    affected_columns = sum(
        1 for column in result.type_validation.results if column.invalid_count > 0
    )

    if total_from_columns != result.type_validation.total_invalid_values:
        return _failed(
            "type_validation_total",
            (
                "Type validation total is inconsistent. "
                f"Reported {result.type_validation.total_invalid_values}, "
                f"column-level sum {total_from_columns}."
            ),
        )

    if affected_columns != result.type_validation.columns_with_invalid_values:
        return _failed(
            "type_validation_affected_columns",
            (
                "Type validation affected-column count is inconsistent. "
                f"Reported {result.type_validation.columns_with_invalid_values}, "
                f"recomputed {affected_columns}."
            ),
        )

    return _passed(
        "type_validation",
        "Type validation totals are internally consistent.",
    )


def _verify_business_rules(result: DataQualityRunResult) -> VerificationCheckResult:
    if result.business_rules is None:
        return _passed(
            "business_rules",
            "Business rules were not configured, so no business-rule consistency check was required.",
        )

    total_from_rules = sum(
        rule.violation_count for rule in result.business_rules.results
    )

    rules_with_violations = sum(
        1 for rule in result.business_rules.results if rule.violation_count > 0
    )

    if total_from_rules != result.business_rules.total_violations:
        return _failed(
            "business_rules_total",
            (
                "Business-rule total is inconsistent. "
                f"Reported {result.business_rules.total_violations}, "
                f"rule-level sum {total_from_rules}."
            ),
        )

    if rules_with_violations != result.business_rules.rules_with_violations:
        return _failed(
            "business_rules_affected_rules",
            (
                "Business-rule affected-rule count is inconsistent. "
                f"Reported {result.business_rules.rules_with_violations}, "
                f"recomputed {rules_with_violations}."
            ),
        )

    return _passed(
        "business_rules",
        "Business-rule totals are internally consistent.",
    )


def _verify_sql_analysis(result: DataQualityRunResult) -> VerificationCheckResult:
    if result.sql_analysis is None:
        return _passed(
            "sql_analysis",
            "SQL analysis was not configured, so no SQL verification was required.",
        )

    for query_result in result.sql_analysis.results:
        try:
            validate_read_only_sql(query_result.sql)
        except Exception as error:
            return _failed(
                "sql_read_only_validation",
                (
                    f"SQL query `{query_result.query_name}` failed read-only validation: "
                    f"{error}"
                ),
            )

    if result.sql_analysis.queries_failed > 0:
        return _warning(
            "sql_execution",
            (
                f"{result.sql_analysis.queries_failed} SQL query/queries failed during execution. "
                "The final report should mention this limitation."
            ),
        )

    return _passed(
        "sql_analysis",
        "All SQL analysis queries are read-only and executed successfully.",
    )


def _verify_bi_readiness_score(result: DataQualityRunResult) -> VerificationCheckResult:
    if result.bi_readiness_score is None:
        return _warning(
            "bi_readiness_score",
            "BI-readiness score was not computed.",
        )

    score = result.bi_readiness_score

    breakdown_total = sum(component.score for component in score.breakdown)

    if breakdown_total != score.overall_score:
        return _failed(
            "bi_readiness_score_total",
            (
                "BI-readiness score is inconsistent. "
                f"Overall score is {score.overall_score}, but breakdown sum is "
                f"{breakdown_total}."
            ),
        )

    invalid_components = [
        component.component_name
        for component in score.breakdown
        if component.score < 0 or component.score > component.max_score
    ]

    if invalid_components:
        return _failed(
            "bi_readiness_score_components",
            f"Some BI-readiness components have invalid scores: {invalid_components}",
        )

    return _passed(
        "bi_readiness_score",
        "BI-readiness score matches the sum of its component scores.",
    )


def verify_data_quality_result(
    dataframe: pd.DataFrame,
    result: DataQualityRunResult,
) -> VerificationRunResult:
    """
    Verify that a data quality run result is internally consistent.

    This is a deterministic verification layer.
    It does not use an LLM.
    """
    checks = [
        _verify_schema_counts(dataframe, result),
        _verify_missing_values(dataframe, result),
        _verify_duplicates(dataframe, result),
        _verify_type_validation(result),
        _verify_business_rules(result),
        _verify_sql_analysis(result),
        _verify_bi_readiness_score(result),
    ]

    checks_failed = sum(1 for check in checks if check.status == "failed")
    checks_warning = sum(1 for check in checks if check.status == "warning")

    if checks_failed > 0:
        status = "failed"
    elif checks_warning > 0:
        status = "warning"
    else:
        status = "passed"

    return VerificationRunResult(
        status=status,
        checks_run=len(checks),
        checks_failed=checks_failed,
        checks_warning=checks_warning,
        results=checks,
    )