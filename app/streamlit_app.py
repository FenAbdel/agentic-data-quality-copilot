from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.agent.copilot import run_copilot_analysis
from dq_copilot.models import (
    BusinessRuleConfig,
    DataQualityRunConfig,
    SQLAnalysisQuery,
)
from dq_copilot.reporting.markdown_report import generate_markdown_report

st.set_page_config(
    page_title="Agentic Data Quality Copilot",
    page_icon="🧪",
    layout="wide",
)

SAMPLE_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "customers.csv"

# Emoji badges keep every status/severity/rating scannable at a glance,
# matching the "traffic light" convention used by tools like Great
# Expectations Data Docs and Evidently reports.
STATUS_STYLES: dict[str, tuple[str, str]] = {
    "passed": ("✅", "Passed"),
    "completed": ("✅", "Completed"),
    "strong": ("✅", "Strong"),
    "excellent": ("✅", "Excellent"),
    "good": ("✅", "Good"),
    "acceptable": ("🟡", "Acceptable"),
    "warning": ("🟡", "Warning"),
    "needs_attention": ("🟠", "Needs attention"),
    "weak": ("🟠", "Weak"),
    "failed": ("🔴", "Failed"),
    "poor": ("🔴", "Poor"),
    "critical": ("🔴", "Critical"),
    "skipped": ("⏭️", "Skipped"),
    "info": ("ℹ️", "Info"),
}

SEVERITY_STYLES: dict[str, tuple[str, str]] = {
    "ok": ("✅", "OK"),
    "low": ("🟡", "Low"),
    "medium": ("🟠", "Medium"),
    "high": ("🔴", "High"),
}


def _badge(value: str, styles: dict[str, tuple[str, str]]) -> str:
    emoji, label = styles.get(value, ("", value.replace("_", " ").title()))
    return f"{emoji} {label}".strip()


def status_badge(value: str) -> str:
    """Render a check status as an emoji + label, e.g. '✅ Passed'."""
    return _badge(value, STATUS_STYLES)


def severity_badge(value: str) -> str:
    """Render a severity level as an emoji + label, e.g. '🔴 High'."""
    return _badge(value, SEVERITY_STYLES)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a Streamlit uploaded CSV file into a DataFrame.

    This function is specific to the UI layer because Streamlit upload objects
    are different from local file paths.
    """
    return pd.read_csv(uploaded_file)


def parse_comma_separated_values(raw_value: str) -> list[str]:
    """
    Convert a comma-separated text input into a clean list of strings.
    """
    if not raw_value.strip():
        return []

    return [value.strip() for value in raw_value.split(",") if value.strip()]


def reset_app_state() -> None:
    """Clear the uploaded file, sample dataset flag, and configured checks."""
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.session_state.use_sample_dataset = False
    st.session_state.business_rules = []
    st.session_state.sql_queries = []
    st.session_state.pop("dataframe", None)
    st.session_state.pop("dataset_name", None)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧪 DQ Copilot")
        st.caption("Agentic data quality assistant")

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Upload a CSV (or try the sample)\n"
            "2. Pick Auto Copilot or Manual mode\n"
            "3. Run the checks\n"
            "4. Review the score and download the report"
        )

        current_dataframe = st.session_state.get("dataframe")
        if current_dataframe is not None:
            st.divider()
            st.markdown("**Current dataset**")
            st.write(f"`{st.session_state.get('dataset_name', 'dataset')}`")
            st.caption(f"{len(current_dataframe):,} rows · {len(current_dataframe.columns)} columns")

            if st.button("🔄 Start over", use_container_width=True):
                reset_app_state()
                st.rerun()

        st.divider()
        st.caption(
            "The LLM plans and explains — Python and DuckDB calculate and "
            "verify every result, so nothing here is free text."
        )


def render_empty_state() -> None:
    st.info("👆 Upload a CSV file above, or click **Try the sample dataset** to explore the app.")

    st.markdown("#### What gets checked")
    feature_columns = st.columns(4)
    features = [
        ("🧬", "Schema & types", "Column dtypes, sample values, and type validation."),
        ("❓", "Missing values", "Null counts and severity, per column."),
        ("🧩", "Duplicates", "Full-row or key-based duplicate detection."),
        ("🎯", "BI-readiness", "A 0-100 score with concrete recommendations."),
    ]
    for column, (icon, title, description) in zip(feature_columns, features):
        with column:
            st.markdown(f"**{icon} {title}**")
            st.caption(description)


def build_expected_schema_from_ui(dataframe: pd.DataFrame) -> dict[str, str]:
    """
    Build an expected schema from simple Streamlit selectors.
    """
    st.subheader("🧬 Expected schema")

    with st.expander("Configure expected column types", expanded=False):
        st.caption(
            "Select expected types for columns you want to validate. "
            "Leave a column as 'skip' if you do not want type validation for it."
        )

        type_options = [
            "skip",
            "integer",
            "float",
            "numeric",
            "date",
            "string",
            "boolean",
        ]

        expected_schema: dict[str, str] = {}

        for column in dataframe.columns:
            selected_type = st.selectbox(
                label=f"Expected type for `{column}`",
                options=type_options,
                index=0,
                key=f"type_{column}",
            )

            if selected_type != "skip":
                expected_schema[str(column)] = selected_type

    return expected_schema


def build_business_rules_from_ui(dataframe: pd.DataFrame) -> list[BusinessRuleConfig]:
    """
    Build a small set of business rules from UI inputs.

    This first UI version supports common rule types only.
    We keep it simple so the app remains understandable.
    """
    if "business_rules" not in st.session_state:
        st.session_state.business_rules = []

    rule_count = len(st.session_state.business_rules)
    expander_label = "Configure business rules"
    if rule_count:
        expander_label += f" — {rule_count} added"

    st.subheader("📏 Business rules")

    with st.expander(expander_label, expanded=False):
        st.caption(
            "Add simple business rules. This first version supports not-null, "
            "positive, non-negative, range, and allowed-values checks."
        )

        selected_column = st.selectbox(
            "Column",
            options=list(dataframe.columns),
            key="business_rule_column",
        )

        rule_type = st.selectbox(
            "Rule type",
            options=[
                "not_null",
                "positive",
                "non_negative",
                "range",
                "allowed_values",
                "not_future_date",
            ],
            key="business_rule_type",
        )

        rule_name = st.text_input(
            "Rule name",
            value=f"{selected_column}_{rule_type}",
            key="business_rule_name",
        )

        allow_null = st.checkbox(
            "Allow null values for this rule",
            value=True,
            key="business_rule_allow_null",
        )

        min_value = None
        max_value = None
        allowed_values = None

        if rule_type == "range":
            col1, col2 = st.columns(2)

            with col1:
                min_value = st.number_input(
                    "Minimum value",
                    value=0.0,
                    key="business_rule_min_value",
                )

            with col2:
                max_value = st.number_input(
                    "Maximum value",
                    value=100.0,
                    key="business_rule_max_value",
                )

        if rule_type == "allowed_values":
            raw_allowed_values = st.text_input(
                "Allowed values",
                placeholder="France, Morocco, Spain",
                key="business_rule_allowed_values",
            )
            allowed_values = parse_comma_separated_values(raw_allowed_values)

        add_rule = st.button("➕ Add business rule")

        if add_rule:
            try:
                rule = BusinessRuleConfig(
                    rule_name=rule_name,
                    column_name=str(selected_column),
                    rule_type=rule_type,
                    min_value=min_value,
                    max_value=max_value,
                    allowed_values=allowed_values,
                    allow_null=allow_null,
                )

                st.session_state.business_rules.append(rule)
                st.success(f"Added rule: {rule.rule_name}")

            except Exception as error:
                st.error(f"Could not add rule: {error}")

        if st.session_state.business_rules:
            st.divider()
            st.write(f"**{len(st.session_state.business_rules)} rule(s) configured:**")

            for index, rule in enumerate(st.session_state.business_rules):
                rule_col, delete_col = st.columns([6, 1])
                with rule_col:
                    st.write(
                        f"`{rule.rule_name}` — column **{rule.column_name}**, "
                        f"type `{rule.rule_type}`"
                    )
                with delete_col:
                    if st.button("🗑️", key=f"delete_rule_{index}", help="Remove this rule"):
                        st.session_state.business_rules.pop(index)
                        st.rerun()

        rules = st.session_state.business_rules

    return rules


def build_sql_queries_from_ui() -> list[SQLAnalysisQuery]:
    if "sql_queries" not in st.session_state:
        st.session_state.sql_queries = []

    query_count = len(st.session_state.sql_queries)
    expander_label = "Configure SQL queries"
    if query_count:
        expander_label += f" — {query_count} added"

    st.subheader("🗄️ DuckDB SQL analysis")

    with st.expander(expander_label, expanded=False):
        st.caption(
            "Write read-only DuckDB SQL queries against the uploaded dataset. "
            "Use the table name `dataset`."
        )

        query_name = st.text_input(
            "Query name",
            value="rows_by_country",
            key="sql_query_name",
        )

        sql = st.text_area(
            "SQL query",
            value=(
                "SELECT country, COUNT(*) AS row_count\n"
                "FROM dataset\n"
                "GROUP BY country\n"
                "ORDER BY row_count DESC"
            ),
            height=160,
            key="sql_query_text",
        )

        if st.button("➕ Add SQL query"):
            try:
                query = SQLAnalysisQuery(
                    query_name=query_name,
                    sql=sql,
                )
                st.session_state.sql_queries.append(query)
                st.success(f"Added SQL query: {query.query_name}")
            except Exception as error:
                st.error(f"Could not add SQL query: {error}")

        if st.session_state.sql_queries:
            st.divider()
            st.write(f"**{len(st.session_state.sql_queries)} quer(y/ies) configured:**")

            for index, query in enumerate(st.session_state.sql_queries):
                query_col, delete_col = st.columns([6, 1])
                with query_col:
                    st.write(f"`{query.query_name}`")
                with delete_col:
                    if st.button("🗑️", key=f"delete_query_{index}", help="Remove this query"):
                        st.session_state.sql_queries.pop(index)
                        st.rerun()

        return st.session_state.sql_queries


def display_schema_summary(result) -> None:
    st.subheader("🧬 Schema summary")

    schema_rows = [
        {
            "Column": column.name,
            "Pandas dtype": column.pandas_dtype,
            "Non-null": column.non_null_count,
            "Missing": column.null_count,
            "Missing %": column.null_percentage,
            "Sample values": ", ".join(column.sample_values),
        }
        for column in result.schema_profile.columns
    ]

    st.dataframe(schema_rows, use_container_width=True)


def display_missing_values(result) -> None:
    st.subheader("❓ Missing values")

    missing = result.missing_values

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_badge(missing.status))
    col2.metric("Total missing values", missing.total_missing_values)
    col3.metric("Columns affected", missing.columns_with_missing_values)

    rows = [
        {
            "Column": column.column_name,
            "Missing count": column.null_count,
            "Missing %": column.null_percentage,
            "Severity": severity_badge(column.severity),
        }
        for column in missing.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_duplicates(result) -> None:
    st.subheader("🧩 Duplicates")

    duplicates = result.duplicates

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_badge(duplicates.status))
    col2.metric("Duplicated rows", duplicates.duplicate_row_count)
    col3.metric("Duplicate %", duplicates.duplicate_percentage)

    st.write(f"Scope: `{duplicates.scope}`")
    st.write(f"Key columns: `{duplicates.key_columns}`")

    if duplicates.duplicate_groups:
        rows = [
            {
                "Duplicate key": group.duplicate_key,
                "Count": group.duplicate_count,
            }
            for group in duplicates.duplicate_groups
        ]

        st.dataframe(rows, use_container_width=True)


def display_type_validation(result) -> None:
    st.subheader("🔡 Type validation")

    if result.type_validation is None:
        st.info("Type validation was skipped because no expected schema was provided.")
        return

    type_validation = result.type_validation

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_badge(type_validation.status))
    col2.metric("Invalid values", type_validation.total_invalid_values)
    col3.metric("Columns affected", type_validation.columns_with_invalid_values)

    rows = [
        {
            "Column": column.column_name,
            "Expected type": column.expected_type,
            "Pandas dtype": column.pandas_dtype,
            "Invalid count": column.invalid_count,
            "Invalid %": column.invalid_percentage,
            "Severity": severity_badge(column.severity),
        }
        for column in type_validation.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_business_rules(result) -> None:
    st.subheader("📏 Business rules")

    if result.business_rules is None:
        st.info("Business rules were skipped because no rules were configured.")
        return

    business_rules = result.business_rules

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_badge(business_rules.status))
    col2.metric("Total violations", business_rules.total_violations)
    col3.metric("Rules with violations", business_rules.rules_with_violations)

    rows = [
        {
            "Rule": rule.rule_name,
            "Column": rule.column_name,
            "Type": rule.rule_type,
            "Status": status_badge(rule.status),
            "Violations": rule.violation_count,
            "Violation %": rule.violation_percentage,
            "Severity": severity_badge(rule.severity),
        }
        for rule in business_rules.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_action_log(result) -> None:
    st.subheader("🧾 Action log")

    rows = [
        {
            "Step": log.step_name,
            "Status": status_badge(log.status),
            "Message": log.message,
        }
        for log in result.action_log
    ]

    st.dataframe(rows, use_container_width=True)


def display_bi_readiness_score(result) -> None:
    st.subheader("🎯 BI-readiness score")

    if result.bi_readiness_score is None:
        st.info("BI-readiness score was not computed.")
        return

    score = result.bi_readiness_score

    col1, col2 = st.columns(2)
    col1.metric("BI-readiness score", f"{score.overall_score}/{score.max_score}")
    col2.metric("Rating", status_badge(score.rating))

    st.write(score.summary)

    rows = [
        {
            "Component": component.component_name,
            "Score": f"{component.score}/{component.max_score}",
            "Status": status_badge(component.status),
            "Explanation": component.explanation,
        }
        for component in score.breakdown
    ]

    st.dataframe(rows, use_container_width=True)

    st.write("Recommendations:")
    for recommendation in score.recommendations:
        st.write(f"- {recommendation}")


def display_sql_analysis(result) -> None:
    st.subheader("🗄️ DuckDB SQL analysis")

    if result.sql_analysis is None:
        st.info("DuckDB SQL analysis was skipped because no SQL queries were configured.")
        return

    sql_analysis = result.sql_analysis

    col1, col2 = st.columns(2)
    col1.metric("Status", status_badge(sql_analysis.status))
    col2.metric("Queries failed", sql_analysis.queries_failed)

    for query_result in sql_analysis.results:
        st.markdown(f"### `{query_result.query_name}`")
        st.code(query_result.sql, language="sql")

        if query_result.status == "failed":
            st.error(query_result.error_message)
            continue

        st.write(f"Rows returned: `{query_result.row_count}`")
        st.dataframe(query_result.rows, use_container_width=True)


def display_agent_plan(planning_result) -> None:
    st.subheader("🧭 Agent plan")

    st.write(planning_result.plan.planning_summary)

    if planning_result.planning_notes:
        st.write("Planning notes:")
        for note in planning_result.planning_notes:
            st.write(f"- {note}")

    plan_rows = [
        {
            "Step": step.step_id,
            "Tool": step.tool_name,
            "Objective": step.objective,
            "Rationale": step.rationale,
            "Optional": step.is_optional,
            "Depends on": step.depends_on,
        }
        for step in planning_result.plan.steps
    ]

    st.dataframe(plan_rows, use_container_width=True)

    with st.expander("Assumptions and risks", expanded=False):
        st.write("Assumptions:")
        for assumption in planning_result.plan.assumptions:
            st.write(f"- {assumption}")

        st.write("Risks:")
        for risk in planning_result.plan.risks:
            st.write(f"- {risk}")


def display_observations(copilot_result) -> None:
    st.subheader("🔎 Copilot observations")

    rows = [
        {
            "Source": observation.source,
            "Severity": status_badge(observation.severity),
            "Message": observation.message,
        }
        for observation in copilot_result.observations
    ]

    st.dataframe(rows, use_container_width=True)


def display_verification(result) -> None:
    st.subheader("✅ Result verification")

    if result.verification_result is None:
        st.info("Result verification was not executed.")
        return

    verification = result.verification_result

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_badge(verification.status))
    col2.metric("Failed checks", verification.checks_failed)
    col3.metric("Warnings", verification.checks_warning)

    rows = [
        {
            "Check": check.check_name,
            "Status": status_badge(check.status),
            "Message": check.message,
        }
        for check in verification.results
    ]

    st.dataframe(rows, use_container_width=True)


def render_checks_at_a_glance(result) -> None:
    """A one-row summary of every check's status, shown before the tabs."""
    checks = [
        ("❓ Missing values", result.missing_values.status if result.missing_values else None),
        ("🧩 Duplicates", result.duplicates.status if result.duplicates else None),
        ("🔡 Types", result.type_validation.status if result.type_validation else None),
        ("📏 Rules", result.business_rules.status if result.business_rules else None),
        ("🗄️ SQL", result.sql_analysis.status if result.sql_analysis else None),
    ]
    checks = [(name, status) for name, status in checks if status is not None]

    if not checks:
        return

    columns = st.columns(len(checks))
    for column, (name, status) in zip(columns, checks):
        column.metric(name, status_badge(status))


def render_score_overview(score) -> None:
    """A headline BI-readiness card shown above the detail tabs."""
    if score is None:
        return

    with st.container(border=True):
        left, right = st.columns([1, 2])

        with left:
            st.metric("BI-readiness score", f"{score.overall_score} / {score.max_score}")
            st.markdown(f"**{status_badge(score.rating)}**")

        with right:
            fraction = score.overall_score / score.max_score if score.max_score else 0
            st.progress(min(max(fraction, 0.0), 1.0))
            st.caption(score.summary)

        if score.recommendations:
            st.markdown("**Top recommendations:**")
            for recommendation in score.recommendations[:3]:
                st.write(f"- {recommendation}")


def display_auto_copilot_result(copilot_result) -> None:
    st.header("🤖 Auto Copilot analysis")

    st.success(copilot_result.execution_summary)

    dq_result = copilot_result.data_quality_result

    render_score_overview(dq_result.bi_readiness_score)
    render_checks_at_a_glance(dq_result)

    (
        tab_plan,
        tab_observations,
        tab_schema,
        tab_missing,
        tab_duplicates,
        tab_types,
        tab_rules,
        tab_sql,
        tab_score,
        tab_verification,
        tab_log,
        tab_report,
    ) = st.tabs(
        [
            "🧭 Agent plan",
            "🔎 Observations",
            "🧬 Schema",
            "❓ Missing values",
            "🧩 Duplicates",
            "🔡 Type validation",
            "📏 Business rules",
            "🗄️ SQL analysis",
            "🎯 BI-readiness",
            "✅ Verification",
            "🧾 Action log",
            "📄 Markdown report",
        ]
    )

    with tab_plan:
        display_agent_plan(copilot_result.planning_result)

    with tab_observations:
        display_observations(copilot_result)

    with tab_schema:
        display_schema_summary(dq_result)

    with tab_missing:
        display_missing_values(dq_result)

    with tab_duplicates:
        display_duplicates(dq_result)

    with tab_types:
        display_type_validation(dq_result)

    with tab_rules:
        display_business_rules(dq_result)

    with tab_sql:
        display_sql_analysis(dq_result)

    with tab_score:
        display_bi_readiness_score(dq_result)

    with tab_verification:
        display_verification(dq_result)

    with tab_log:
        display_action_log(dq_result)

    with tab_report:
        st.markdown(copilot_result.markdown_report)

        st.download_button(
            label="⬇️ Download Auto Copilot Markdown report",
            data=copilot_result.markdown_report,
            file_name=f"{copilot_result.dataset_name}_auto_copilot_report.md",
            mime="text/markdown",
        )


def main() -> None:
    render_sidebar()

    st.title("🧪 Agentic Data Quality Copilot")
    st.write(
        "Upload a CSV dataset, configure optional checks, and generate a data "
        "quality report using deterministic Python tools."
    )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "use_sample_dataset" not in st.session_state:
        st.session_state.use_sample_dataset = False

    upload_col, sample_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key=f"uploader_{st.session_state.uploader_key}",
        )

    with sample_col:
        st.write("")
        st.write("")
        if st.button("✨ Try the sample dataset", use_container_width=True):
            st.session_state.use_sample_dataset = True
            st.rerun()

    if uploaded_file is not None:
        st.session_state.use_sample_dataset = False

    if uploaded_file is None and not st.session_state.use_sample_dataset:
        render_empty_state()
        return

    try:
        if st.session_state.use_sample_dataset:
            dataset_name = SAMPLE_DATASET_PATH.name
            dataframe = pd.read_csv(SAMPLE_DATASET_PATH)
        else:
            dataset_name = uploaded_file.name
            dataframe = load_uploaded_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read CSV file: {error}")
        return

    st.session_state.dataframe = dataframe
    st.session_state.dataset_name = dataset_name

    st.success(f"Loaded `{dataset_name}` successfully.")

    with st.container(border=True):
        st.subheader("Dataset preview")

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Rows", f"{len(dataframe):,}")
        stat_col2.metric("Columns", len(dataframe.columns))
        stat_col3.metric(
            "Memory",
            f"{dataframe.memory_usage(deep=True).sum() / 1024:.1f} KB",
        )

        st.dataframe(dataframe.head(20), use_container_width=True)

    st.divider()

    st.markdown("### Choose analysis mode")
    analysis_mode = st.radio(
        "Analysis mode",
        options=[
            "Auto Copilot mode",
            "Manual configuration mode",
        ],
        captions=[
            "Recommended — the copilot plans and runs a full check suite for you.",
            "Pick exactly which checks, rules, and SQL queries to run.",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if analysis_mode == "Auto Copilot mode":
        st.header("🤖 Auto Copilot configuration")

        user_goal = st.text_area(
            "Copilot goal",
            value="Assess whether this dataset is ready for BI reporting.",
            help=(
                "The deterministic copilot will use this goal to create a structured plan. "
                "In this version, the goal is used for planning context, not LLM reasoning yet."
            ),
        )

        run_auto_copilot = st.button(
            "▶️ Run Auto Copilot Analysis",
            type="primary",
        )

        if run_auto_copilot:
            try:
                with st.spinner("Running Auto Copilot analysis..."):
                    copilot_result = run_copilot_analysis(
                        dataframe=dataframe,
                        dataset_name=dataset_name,
                        user_goal=user_goal,
                    )

                display_auto_copilot_result(copilot_result)

            except Exception as error:
                st.error(f"Auto Copilot analysis failed: {error}")

        return

    st.header("⚙️ Check configuration")

    duplicate_key_columns = st.multiselect(
        "Duplicate key columns",
        options=list(dataframe.columns),
        help=(
            "Select columns that should uniquely identify a row. "
            "Leave empty to check full-row duplicates."
        ),
    )

    expected_schema = build_expected_schema_from_ui(dataframe)
    business_rules = build_business_rules_from_ui(dataframe)
    sql_queries = build_sql_queries_from_ui()
    st.divider()

    run_checks = st.button("▶️ Run data quality checks", type="primary")

    if not run_checks:
        return

    config = DataQualityRunConfig(
        dataset_name=dataset_name,
        duplicate_key_columns=duplicate_key_columns or None,
        expected_schema=expected_schema or None,
        business_rules=business_rules,
        sql_queries=sql_queries,
    )
    try:
        result = run_data_quality_checks(
            dataframe=dataframe,
            config=config,
        )
    except Exception as error:
        st.error(f"Data quality run failed: {error}")
        return

    report = generate_markdown_report(result)

    st.header("📊 Data quality results")

    render_score_overview(result.bi_readiness_score)
    render_checks_at_a_glance(result)

    (
        tab_schema,
        tab_missing,
        tab_duplicates,
        tab_types,
        tab_rules,
        tab_sql,
        tab_score,
        tab_log,
        tab_report,
    ) = st.tabs(
        [
            "🧬 Schema",
            "❓ Missing values",
            "🧩 Duplicates",
            "🔡 Type validation",
            "📏 Business rules",
            "🗄️ SQL analysis",
            "🎯 BI-readiness",
            "🧾 Action log",
            "📄 Markdown report",
        ]
    )

    with tab_schema:
        display_schema_summary(result)

    with tab_missing:
        display_missing_values(result)

    with tab_duplicates:
        display_duplicates(result)

    with tab_types:
        display_type_validation(result)

    with tab_rules:
        display_business_rules(result)

    with tab_score:
        display_bi_readiness_score(result)

    with tab_log:
        display_action_log(result)

    with tab_sql:
        display_sql_analysis(result)

    with tab_report:
        st.markdown(report)

        st.download_button(
            label="⬇️ Download Markdown report",
            data=report,
            file_name=f"{dataset_name}_quality_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
