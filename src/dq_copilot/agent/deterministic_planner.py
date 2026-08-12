from __future__ import annotations

import re

from dq_copilot.agent.plan_templates import create_default_bi_readiness_plan
from dq_copilot.agent.planning_models import AgentPlanningResult
from dq_copilot.models import (
    BusinessRuleConfig,
    ColumnSchemaProfile,
    DataQualityRunConfig,
    DatasetSchemaProfile,
    ExpectedDataType,
    SQLAnalysisQuery,
)


def _normalize_column_name(column_name: str) -> str:
    """
    Normalize a column name for heuristic matching.
    """
    normalized = column_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _tokens(column_name: str) -> set[str]:
    return set(_normalize_column_name(column_name).split("_"))


def _quote_identifier(identifier: str) -> str:
    """
    Quote a SQL identifier for DuckDB.
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _is_id_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    tokens = _tokens(column_name)

    return normalized == "id" or "id" in tokens or normalized.endswith("_identifier")


def _is_date_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    tokens = _tokens(column_name)

    return bool(
        tokens.intersection({"date", "datetime", "timestamp", "time"})
        or normalized.endswith("_at")
    )


def _is_age_column(column_name: str) -> bool:
    return "age" in _tokens(column_name)


def _is_amount_column(column_name: str) -> bool:
    tokens = _tokens(column_name)

    return bool(
        tokens.intersection(
            {
                "amount",
                "price",
                "cost",
                "revenue",
                "sales",
                "total",
                "balance",
                "value",
            }
        )
    )


def _is_count_column(column_name: str) -> bool:
    tokens = _tokens(column_name)

    return bool(tokens.intersection({"quantity", "qty", "count", "number", "nb"}))


def _is_boolean_like_column(column: ColumnSchemaProfile) -> bool:
    normalized = _normalize_column_name(column.name)
    tokens = _tokens(column.name)

    if column.pandas_dtype.lower() == "bool":
        return True

    if normalized.startswith(("is_", "has_", "can_")):
        return True

    if tokens.intersection({"active", "enabled", "flag"}):
        return True

    accepted_boolean_values = {"true", "false", "1", "0", "yes", "no", "y", "n"}
    sample_values = {value.strip().lower() for value in column.sample_values}

    return bool(sample_values) and sample_values.issubset(accepted_boolean_values)


def _is_categorical_column(column: ColumnSchemaProfile) -> bool:
    tokens = _tokens(column.name)

    if _is_date_column(column.name):
        return False

    if tokens.intersection(
        {
            "country",
            "city",
            "region",
            "status",
            "category",
            "segment",
            "type",
        }
    ):
        return True

    return column.pandas_dtype.lower() == "object" and len(column.sample_values) <= 10


def infer_expected_type(column: ColumnSchemaProfile) -> ExpectedDataType:
    """
    Infer an expected type from column name, Pandas dtype, and sample values.

    This is a heuristic, not a guarantee.
    """
    dtype = column.pandas_dtype.lower()

    if _is_date_column(column.name):
        return "date"

    if _is_age_column(column.name):
        return "integer"

    if _is_boolean_like_column(column):
        return "boolean"

    if _is_amount_column(column.name):
        return "numeric"

    if _is_count_column(column.name):
        return "integer"

    if "int" in dtype:
        return "integer"

    if "float" in dtype:
        return "numeric"

    if "bool" in dtype:
        return "boolean"

    if "datetime" in dtype or "date" in dtype:
        return "date"

    return "string"


def infer_duplicate_key_columns(
    schema_profile: DatasetSchemaProfile,
) -> list[str] | None:
    """
    Infer a likely duplicate-key column from the schema.

    For now, we only infer one strong key candidate.
    Composite key inference can come later.
    """
    for column in schema_profile.columns:
        if _is_id_column(column.name):
            return [column.name]

    return None


def infer_expected_schema(
    schema_profile: DatasetSchemaProfile,
) -> dict[str, ExpectedDataType]:
    """
    Infer expected types for all columns.
    """
    return {
        column.name: infer_expected_type(column)
        for column in schema_profile.columns
    }


def infer_business_rules(
    schema_profile: DatasetSchemaProfile,
) -> list[BusinessRuleConfig]:
    """
    Infer simple business rules from column names.

    We intentionally avoid auto-generating allowed_values rules from samples,
    because sample values may not represent the full valid business domain.
    """
    rules: list[BusinessRuleConfig] = []
    used_rule_names: set[str] = set()

    def add_rule(rule: BusinessRuleConfig) -> None:
        if rule.rule_name not in used_rule_names:
            used_rule_names.add(rule.rule_name)
            rules.append(rule)

    for column in schema_profile.columns:
        normalized_name = _normalize_column_name(column.name)

        if _is_id_column(column.name):
            add_rule(
                BusinessRuleConfig(
                    rule_name=f"{normalized_name}_required",
                    column_name=column.name,
                    rule_type="not_null",
                    allow_null=False,
                    description="Identifier columns are usually required for joins and traceability.",
                )
            )

        if _is_age_column(column.name):
            add_rule(
                BusinessRuleConfig(
                    rule_name=f"{normalized_name}_valid_range",
                    column_name=column.name,
                    rule_type="range",
                    min_value=0,
                    max_value=120,
                    allow_null=True,
                    description="Age values should be within a realistic human range.",
                )
            )

        if _is_amount_column(column.name):
            add_rule(
                BusinessRuleConfig(
                    rule_name=f"{normalized_name}_non_negative",
                    column_name=column.name,
                    rule_type="non_negative",
                    allow_null=True,
                    description="Financial or amount-like values are usually expected to be non-negative.",
                )
            )

        if _is_date_column(column.name):
            add_rule(
                BusinessRuleConfig(
                    rule_name=f"{normalized_name}_not_future",
                    column_name=column.name,
                    rule_type="not_future_date",
                    allow_null=True,
                    description="Date-like columns should usually not contain future dates.",
                )
            )

    return rules


def infer_sql_queries(
    schema_profile: DatasetSchemaProfile,
    max_categorical_queries: int = 3,
    max_numeric_queries: int = 3,
    max_date_queries: int = 2,
) -> list[SQLAnalysisQuery]:
    """
    Infer useful read-only DuckDB SQL analysis queries.
    """
    queries: list[SQLAnalysisQuery] = [
        SQLAnalysisQuery(
            query_name="total_rows",
            sql="SELECT COUNT(*) AS row_count FROM dataset",
            description="Count total rows in the dataset.",
        )
    ]

    categorical_columns = [
        column for column in schema_profile.columns if _is_categorical_column(column)
    ]

    for column in categorical_columns[:max_categorical_queries]:
        quoted_column = _quote_identifier(column.name)
        normalized_name = _normalize_column_name(column.name)

        queries.append(
            SQLAnalysisQuery(
                query_name=f"{normalized_name}_distribution",
                sql=(
                    f"SELECT {quoted_column} AS value, COUNT(*) AS row_count\n"
                    "FROM dataset\n"
                    f"GROUP BY {quoted_column}\n"
                    "ORDER BY row_count DESC"
                ),
                description=f"Analyze value distribution for `{column.name}`.",
            )
        )

    numeric_columns = [
        column
        for column in schema_profile.columns
        if infer_expected_type(column) in {"integer", "float", "numeric"}
    ]

    for column in numeric_columns[:max_numeric_queries]:
        quoted_column = _quote_identifier(column.name)
        normalized_name = _normalize_column_name(column.name)

        queries.append(
            SQLAnalysisQuery(
                query_name=f"{normalized_name}_numeric_summary",
                sql=(
                    f"SELECT\n"
                    f"  MIN(TRY_CAST({quoted_column} AS DOUBLE)) AS min_value,\n"
                    f"  MAX(TRY_CAST({quoted_column} AS DOUBLE)) AS max_value,\n"
                    f"  AVG(TRY_CAST({quoted_column} AS DOUBLE)) AS avg_value\n"
                    "FROM dataset"
                ),
                description=f"Compute numeric summary for `{column.name}`.",
            )
        )

    date_columns = [
        column for column in schema_profile.columns if _is_date_column(column.name)
    ]

    for column in date_columns[:max_date_queries]:
        quoted_column = _quote_identifier(column.name)
        normalized_name = _normalize_column_name(column.name)

        queries.append(
            SQLAnalysisQuery(
                query_name=f"{normalized_name}_date_range",
                sql=(
                    f"SELECT\n"
                    f"  MIN(TRY_CAST({quoted_column} AS DATE)) AS min_date,\n"
                    f"  MAX(TRY_CAST({quoted_column} AS DATE)) AS max_date\n"
                    "FROM dataset"
                ),
                description=f"Compute date range for `{column.name}`.",
            )
        )

    return queries


def create_deterministic_run_config(
    schema_profile: DatasetSchemaProfile,
) -> DataQualityRunConfig:
    """
    Create a runnable data quality configuration from a schema profile.
    """
    return DataQualityRunConfig(
        dataset_name=schema_profile.dataset_name,
        duplicate_key_columns=infer_duplicate_key_columns(schema_profile),
        expected_schema=infer_expected_schema(schema_profile),
        business_rules=infer_business_rules(schema_profile),
        sql_queries=infer_sql_queries(schema_profile),
    )


def create_deterministic_planning_result(
    schema_profile: DatasetSchemaProfile,
    user_goal: str = "Assess whether this dataset is ready for BI reporting.",
) -> AgentPlanningResult:
    """
    Create a deterministic planning result containing both the plan and run config.
    """
    plan = create_default_bi_readiness_plan(
        dataset_name=schema_profile.dataset_name,
        user_goal=user_goal,
    )

    run_config = create_deterministic_run_config(schema_profile)

    planning_notes: list[str] = []

    if run_config.duplicate_key_columns:
        planning_notes.append(
            f"Detected likely key column(s): {run_config.duplicate_key_columns}."
        )
    else:
        planning_notes.append(
            "No obvious key column was detected, so duplicate checks will use full-row duplicates."
        )

    planning_notes.append(
        f"Inferred expected types for {len(run_config.expected_schema or {})} columns."
    )

    if run_config.business_rules:
        planning_notes.append(
            f"Inferred {len(run_config.business_rules)} business rule(s) from column names."
        )
    else:
        planning_notes.append(
            "No business rules were inferred from the available column names."
        )

    if run_config.sql_queries:
        planning_notes.append(
            f"Generated {len(run_config.sql_queries)} read-only DuckDB SQL analysis query/queries."
        )

    return AgentPlanningResult(
        user_goal=user_goal,
        dataset_name=schema_profile.dataset_name,
        plan=plan,
        run_config=run_config,
        planning_notes=planning_notes,
    )