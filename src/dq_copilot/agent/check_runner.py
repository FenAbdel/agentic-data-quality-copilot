import pandas as pd

from dq_copilot.checks.business_rules import check_business_rules
from dq_copilot.checks.duplicates import check_duplicates
from dq_copilot.checks.missing_values import check_missing_values
from dq_copilot.checks.type_validation import check_type_validation
from dq_copilot.models import (
    CheckExecutionLog,
    DataQualityRunConfig,
    DataQualityRunResult,
)
from dq_copilot.profiling.schema_profiler import profile_schema
from dq_copilot.scoring.bi_readiness import compute_bi_readiness_score
from dq_copilot.sql.duckdb_analyzer import run_duckdb_analysis

def _log_completed(step_name: str, message: str) -> CheckExecutionLog:
    return CheckExecutionLog(
        step_name=step_name,
        status="completed",
        message=message,
    )


def _log_skipped(step_name: str, message: str) -> CheckExecutionLog:
    return CheckExecutionLog(
        step_name=step_name,
        status="skipped",
        message=message,
    )


def run_data_quality_checks(
    dataframe: pd.DataFrame,
    config: DataQualityRunConfig | None = None,
) -> DataQualityRunResult:
    """
    Run the deterministic data quality workflow.

    This function is not an LLM agent.
    It is the deterministic execution layer that the future agent will call.

    Parameters
    ----------
    dataframe:
        Dataset to analyze.

    config:
        Data quality run configuration.

    Returns
    -------
    DataQualityRunResult
        Unified result containing all executed checks and action logs.
    """
    if config is None:
        config = DataQualityRunConfig()

    action_log: list[CheckExecutionLog] = []

    schema_profile = profile_schema(
        dataframe=dataframe,
        dataset_name=config.dataset_name,
    )
    action_log.append(
        _log_completed(
            step_name="schema_profile",
            message=(
                f"Profiled dataset with {schema_profile.row_count} rows "
                f"and {schema_profile.column_count} columns."
            ),
        )
    )

    missing_values = check_missing_values(dataframe)
    action_log.append(
        _log_completed(
            step_name="missing_values",
            message=(
                f"Found {missing_values.total_missing_values} missing values "
                f"across {missing_values.columns_with_missing_values} columns."
            ),
        )
    )

    duplicates = check_duplicates(
        dataframe=dataframe,
        key_columns=config.duplicate_key_columns,
    )

    if config.duplicate_key_columns:
        duplicate_message = (
            f"Checked duplicates using key columns: {config.duplicate_key_columns}. "
            f"Found {duplicates.duplicate_row_count} duplicated rows."
        )
    else:
        duplicate_message = (
            "Checked full-row duplicates. "
            f"Found {duplicates.duplicate_row_count} duplicated rows."
        )

    action_log.append(
        _log_completed(
            step_name="duplicates",
            message=duplicate_message,
        )
    )

    type_validation = None

    if config.expected_schema:
        type_validation = check_type_validation(
            dataframe=dataframe,
            expected_schema=config.expected_schema,
        )
        action_log.append(
            _log_completed(
                step_name="type_validation",
                message=(
                    f"Validated {type_validation.columns_checked} columns against "
                    f"the expected schema and found "
                    f"{type_validation.total_invalid_values} invalid values."
                ),
            )
        )
    else:
        action_log.append(
            _log_skipped(
                step_name="type_validation",
                message="Skipped type validation because no expected schema was provided.",
            )
        )

    business_rules = None

    if config.business_rules:
        business_rules = check_business_rules(
            dataframe=dataframe,
            rules=config.business_rules,
        )
        action_log.append(
            _log_completed(
                step_name="business_rules",
                message=(
                    f"Executed {business_rules.rules_checked} business rules and found "
                    f"{business_rules.total_violations} violations."
                ),
            )
        )
    else:
        action_log.append(
            _log_skipped(
                step_name="business_rules",
                message="Skipped business-rule checks because no rules were configured.",
            )
        )

    sql_analysis = None

    if config.sql_queries:
        sql_analysis = run_duckdb_analysis(
            dataframe=dataframe,
            queries=config.sql_queries,
            table_name="dataset",
        )
        action_log.append(
            _log_completed(
                step_name="duckdb_sql_analysis",
                message=(
                    f"Executed {sql_analysis.queries_executed} DuckDB SQL queries "
                    f"with {sql_analysis.queries_failed} failures."
                ),
            )
        )
    else:
        action_log.append(
            _log_skipped(
                step_name="duckdb_sql_analysis",
                message="Skipped DuckDB SQL analysis because no SQL queries were configured.",
            )
        )

    run_result = DataQualityRunResult(
        dataset_name=config.dataset_name,
        schema_profile=schema_profile,
        missing_values=missing_values,
        duplicates=duplicates,
        type_validation=type_validation,
        business_rules=business_rules,
        sql_analysis=sql_analysis,
        action_log=action_log,
    )

    run_result.bi_readiness_score = compute_bi_readiness_score(run_result)

    action_log.append(
        _log_completed(
            step_name="bi_readiness_score",
            message=(
                f"Computed BI-readiness score: "
                f"{run_result.bi_readiness_score.overall_score}/100 "
                f"({run_result.bi_readiness_score.rating})."
            ),
        )
    )

    return run_result