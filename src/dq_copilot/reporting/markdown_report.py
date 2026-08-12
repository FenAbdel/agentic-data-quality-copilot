from pathlib import Path

from dq_copilot.models import DataQualityRunResult


def _status_icon(status: str) -> str:
    """
    Return a simple icon for a check status.
    """
    if status == "passed":
        return "✅"

    if status == "warning":
        return "⚠️"

    if status == "failed":
        return "❌"

    if status == "completed":
        return "✅"

    if status == "skipped":
        return "⏭️"

    return "ℹ️"


def _format_percentage(value: float) -> str:
    """
    Format percentage values consistently.
    """
    return f"{value:.2f}%"


def _build_dataset_overview_section(result: DataQualityRunResult) -> list[str]:
    schema = result.schema_profile

    return [
        "## Dataset overview",
        "",
        f"- **Dataset name:** `{result.dataset_name}`",
        f"- **Rows:** {schema.row_count}",
        f"- **Columns:** {schema.column_count}",
        "",
    ]


def _build_schema_section(result: DataQualityRunResult) -> list[str]:
    lines = [
        "## Schema summary",
        "",
        "| Column | Pandas dtype | Non-null | Missing | Missing % | Sample values |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for column in result.schema_profile.columns:
        sample_values = ", ".join(column.sample_values)
        lines.append(
            "| "
            f"{column.name} | "
            f"{column.pandas_dtype} | "
            f"{column.non_null_count} | "
            f"{column.null_count} | "
            f"{_format_percentage(column.null_percentage)} | "
            f"{sample_values} |"
        )

    lines.append("")
    return lines


def _build_missing_values_section(result: DataQualityRunResult) -> list[str]:
    missing = result.missing_values

    lines = [
        "## Missing value analysis",
        "",
        f"**Status:** {_status_icon(missing.status)} `{missing.status}`",
        "",
        f"- **Total missing values:** {missing.total_missing_values}",
        f"- **Columns checked:** {missing.columns_checked}",
        f"- **Columns with missing values:** {missing.columns_with_missing_values}",
        "",
        "| Column | Missing count | Missing % | Severity |",
        "|---|---:|---:|---|",
    ]

    for column in missing.results:
        lines.append(
            "| "
            f"{column.column_name} | "
            f"{column.null_count} | "
            f"{_format_percentage(column.null_percentage)} | "
            f"{column.severity} |"
        )

    lines.append("")
    return lines


def _build_duplicates_section(result: DataQualityRunResult) -> list[str]:
    duplicates = result.duplicates

    lines = [
        "## Duplicate detection",
        "",
        f"**Status:** {_status_icon(duplicates.status)} `{duplicates.status}`",
        "",
        f"- **Scope:** `{duplicates.scope}`",
        f"- **Key columns:** {', '.join(duplicates.key_columns) if duplicates.key_columns else 'None'}",
        f"- **Rows checked:** {duplicates.row_count}",
        f"- **Duplicated rows:** {duplicates.duplicate_row_count}",
        f"- **Duplicate percentage:** {_format_percentage(duplicates.duplicate_percentage)}",
        f"- **Duplicate groups:** {duplicates.duplicate_group_count}",
        f"- **Severity:** `{duplicates.severity}`",
        "",
    ]

    if duplicates.duplicate_groups:
        lines.extend(
            [
                "### Duplicate examples",
                "",
                "| Duplicate key | Count |",
                "|---|---:|",
            ]
        )

        for group in duplicates.duplicate_groups:
            duplicate_key = ", ".join(
                f"{key}={value}" for key, value in group.duplicate_key.items()
            )
            lines.append(f"| {duplicate_key} | {group.duplicate_count} |")

        lines.append("")

    return lines


def _build_type_validation_section(result: DataQualityRunResult) -> list[str]:
    if result.type_validation is None:
        return [
            "## Type validation",
            "",
            "⏭️ Type validation was skipped because no expected schema was provided.",
            "",
        ]

    type_validation = result.type_validation

    lines = [
        "## Type validation",
        "",
        f"**Status:** {_status_icon(type_validation.status)} `{type_validation.status}`",
        "",
        f"- **Columns checked:** {type_validation.columns_checked}",
        f"- **Columns with invalid values:** {type_validation.columns_with_invalid_values}",
        f"- **Total invalid values:** {type_validation.total_invalid_values}",
        "",
        "| Column | Expected type | Pandas dtype | Non-null | Invalid count | Invalid % | Severity |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for column in type_validation.results:
        lines.append(
            "| "
            f"{column.column_name} | "
            f"{column.expected_type} | "
            f"{column.pandas_dtype} | "
            f"{column.non_null_count} | "
            f"{column.invalid_count} | "
            f"{_format_percentage(column.invalid_percentage)} | "
            f"{column.severity} |"
        )

    invalid_examples = [
        (column.column_name, example)
        for column in type_validation.results
        for example in column.invalid_examples
    ]

    if invalid_examples:
        lines.extend(
            [
                "",
                "### Invalid value examples",
                "",
                "| Column | Row index | Value | Expected type |",
                "|---|---:|---|---|",
            ]
        )

        for column_name, example in invalid_examples:
            lines.append(
                "| "
                f"{column_name} | "
                f"{example.row_index} | "
                f"{example.value} | "
                f"{example.expected_type} |"
            )

    lines.append("")
    return lines


def _build_business_rules_section(result: DataQualityRunResult) -> list[str]:
    if result.business_rules is None:
        return [
            "## Business-rule checks",
            "",
            "⏭️ Business-rule checks were skipped because no rules were configured.",
            "",
        ]

    business_rules = result.business_rules

    lines = [
        "## Business-rule checks",
        "",
        f"**Status:** {_status_icon(business_rules.status)} `{business_rules.status}`",
        "",
        f"- **Rules checked:** {business_rules.rules_checked}",
        f"- **Rules with violations:** {business_rules.rules_with_violations}",
        f"- **Total violations:** {business_rules.total_violations}",
        "",
        "| Rule | Column | Type | Status | Violations | Violation % | Severity |",
        "|---|---|---|---|---:|---:|---|",
    ]

    for rule in business_rules.results:
        lines.append(
            "| "
            f"{rule.rule_name} | "
            f"{rule.column_name} | "
            f"{rule.rule_type} | "
            f"{_status_icon(rule.status)} {rule.status} | "
            f"{rule.violation_count} | "
            f"{_format_percentage(rule.violation_percentage)} | "
            f"{rule.severity} |"
        )

    violation_examples = [
        (rule.rule_name, rule.column_name, example)
        for rule in business_rules.results
        for example in rule.violation_examples
    ]

    if violation_examples:
        lines.extend(
            [
                "",
                "### Business-rule violation examples",
                "",
                "| Rule | Column | Row index | Value | Message |",
                "|---|---|---:|---|---|",
            ]
        )

        for rule_name, column_name, example in violation_examples:
            lines.append(
                "| "
                f"{rule_name} | "
                f"{column_name} | "
                f"{example.row_index} | "
                f"{example.value} | "
                f"{example.message} |"
            )

    lines.append("")
    return lines

def _build_sql_analysis_section(result: DataQualityRunResult) -> list[str]:
    if result.sql_analysis is None:
        return [
            "## DuckDB SQL analysis",
            "",
            "⏭️ DuckDB SQL analysis was skipped because no SQL queries were configured.",
            "",
        ]

    sql_analysis = result.sql_analysis

    lines = [
        "## DuckDB SQL analysis",
        "",
        f"**Status:** {_status_icon(sql_analysis.status)} `{sql_analysis.status}`",
        "",
        f"- **Queries executed:** {sql_analysis.queries_executed}",
        f"- **Queries failed:** {sql_analysis.queries_failed}",
        "",
    ]

    for query_result in sql_analysis.results:
        lines.extend(
            [
                f"### Query: `{query_result.query_name}`",
                "",
                "```sql",
                query_result.sql.strip(),
                "```",
                "",
                f"**Status:** {_status_icon(query_result.status)} `{query_result.status}`",
                "",
            ]
        )

        if query_result.status == "failed":
            lines.extend(
                [
                    f"**Error:** {query_result.error_message}",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                f"**Rows returned:** {query_result.row_count}",
                "",
            ]
        )

        if query_result.rows:
            header = "| " + " | ".join(query_result.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(query_result.columns)) + " |"

            lines.append(header)
            lines.append(separator)

            for row in query_result.rows:
                lines.append(
                    "| "
                    + " | ".join(row.get(column, "") for column in query_result.columns)
                    + " |"
                )

            lines.append("")

    return lines

def _build_bi_readiness_section(result: DataQualityRunResult) -> list[str]:
    if result.bi_readiness_score is None:
        return [
            "## BI-readiness score",
            "",
            "BI-readiness score was not computed.",
            "",
        ]

    score = result.bi_readiness_score

    lines = [
        "## BI-readiness score",
        "",
        f"**Overall score:** {score.overall_score}/{score.max_score}",
        f"**Rating:** `{score.rating}`",
        "",
        score.summary,
        "",
        "| Component | Score | Status | Explanation |",
        "|---|---:|---|---|",
    ]

    for component in score.breakdown:
        lines.append(
            "| "
            f"{component.component_name} | "
            f"{component.score}/{component.max_score} | "
            f"{component.status} | "
            f"{component.explanation} |"
        )

    lines.append("")
    return lines

def _build_verification_section(result: DataQualityRunResult) -> list[str]:
    if result.verification_result is None:
        return [
            "## Result verification",
            "",
            "Result verification was not executed.",
            "",
        ]

    verification = result.verification_result

    lines = [
        "## Result verification",
        "",
        f"**Status:** {_status_icon(verification.status)} `{verification.status}`",
        "",
        f"- **Checks run:** {verification.checks_run}",
        f"- **Checks failed:** {verification.checks_failed}",
        f"- **Checks with warnings:** {verification.checks_warning}",
        "",
        "| Check | Status | Message |",
        "|---|---|---|",
    ]

    for check in verification.results:
        lines.append(
            "| "
            f"{check.check_name} | "
            f"{_status_icon(check.status)} {check.status} | "
            f"{check.message} |"
        )

    lines.append("")
    return lines


def _build_action_log_section(result: DataQualityRunResult) -> list[str]:
    lines = [
        "## Action log",
        "",
        "| Step | Status | Message |",
        "|---|---|---|",
    ]

    for log in result.action_log:
        lines.append(
            "| "
            f"{log.step_name} | "
            f"{_status_icon(log.status)} {log.status} | "
            f"{log.message} |"
        )

    lines.append("")
    return lines


def _build_recommendations_section(result: DataQualityRunResult) -> list[str]:
    lines = [
        "## Recommendations",
        "",
    ]

    recommendations: list[str] = []

    if result.missing_values.total_missing_values > 0:
        recommendations.append(
            "Review columns with missing values and decide whether to fill, exclude, or correct them at the source."
        )

    if result.duplicates.duplicate_row_count > 0:
        recommendations.append(
            "Investigate duplicated rows or duplicated keys because they may inflate BI metrics or break joins."
        )

    if result.type_validation and result.type_validation.total_invalid_values > 0:
        recommendations.append(
            "Fix invalid type values before loading the dataset into downstream analytical models."
        )

    if result.business_rules and result.business_rules.total_violations > 0:
        recommendations.append(
            "Review business-rule violations with business stakeholders to confirm the expected data contract."
        )

    if not recommendations:
        recommendations.append(
            "No major data quality issue was detected by the configured checks."
        )

    for recommendation in recommendations:
        lines.append(f"- {recommendation}")

    lines.append("")
    return lines


def generate_markdown_report(result: DataQualityRunResult) -> str:
    """
    Generate a Markdown data quality report from a full run result.
    """
    lines = [
        f"# Data Quality Report — {result.dataset_name}",
        "",
    ]

    lines.extend(_build_dataset_overview_section(result))
    lines.extend(_build_schema_section(result))
    lines.extend(_build_missing_values_section(result))
    lines.extend(_build_duplicates_section(result))
    lines.extend(_build_type_validation_section(result))
    lines.extend(_build_business_rules_section(result))
    lines.extend(_build_sql_analysis_section(result))
    lines.extend(_build_bi_readiness_section(result))
    lines.extend(_build_verification_section(result))
    lines.extend(_build_action_log_section(result))
    lines.extend(_build_recommendations_section(result))

    return "\n".join(lines)


def save_markdown_report(report: str, output_path: str | Path) -> Path:
    """
    Save a Markdown report to disk.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    return path