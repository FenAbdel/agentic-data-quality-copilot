from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.models import BusinessRuleConfig, DataQualityRunConfig
from dq_copilot.reporting.markdown_report import generate_markdown_report


st.set_page_config(
    page_title="Agentic Data Quality Copilot",
    page_icon="🧪",
    layout="wide",
)


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


def build_expected_schema_from_ui(dataframe: pd.DataFrame) -> dict[str, str]:
    """
    Build an expected schema from simple Streamlit selectors.
    """
    st.subheader("Expected schema")

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
    st.subheader("Business rules")

    rules: list[BusinessRuleConfig] = []

    with st.expander("Configure business rules", expanded=False):
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

        add_rule = st.button("Add business rule")

        if "business_rules" not in st.session_state:
            st.session_state.business_rules = []

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
            st.write("Configured rules:")

            for index, rule in enumerate(st.session_state.business_rules, start=1):
                st.write(
                    f"{index}. `{rule.rule_name}` — "
                    f"`{rule.column_name}` / `{rule.rule_type}`"
                )

            if st.button("Clear business rules"):
                st.session_state.business_rules = []
                st.rerun()

        rules = st.session_state.business_rules

    return rules


def display_schema_summary(result) -> None:
    st.subheader("Schema summary")

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
    st.subheader("Missing values")

    missing = result.missing_values

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", missing.status)
    col2.metric("Total missing values", missing.total_missing_values)
    col3.metric("Columns affected", missing.columns_with_missing_values)

    rows = [
        {
            "Column": column.column_name,
            "Missing count": column.null_count,
            "Missing %": column.null_percentage,
            "Severity": column.severity,
        }
        for column in missing.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_duplicates(result) -> None:
    st.subheader("Duplicates")

    duplicates = result.duplicates

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", duplicates.status)
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
    st.subheader("Type validation")

    if result.type_validation is None:
        st.info("Type validation was skipped because no expected schema was provided.")
        return

    type_validation = result.type_validation

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", type_validation.status)
    col2.metric("Invalid values", type_validation.total_invalid_values)
    col3.metric("Columns affected", type_validation.columns_with_invalid_values)

    rows = [
        {
            "Column": column.column_name,
            "Expected type": column.expected_type,
            "Pandas dtype": column.pandas_dtype,
            "Invalid count": column.invalid_count,
            "Invalid %": column.invalid_percentage,
            "Severity": column.severity,
        }
        for column in type_validation.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_business_rules(result) -> None:
    st.subheader("Business rules")

    if result.business_rules is None:
        st.info("Business rules were skipped because no rules were configured.")
        return

    business_rules = result.business_rules

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", business_rules.status)
    col2.metric("Total violations", business_rules.total_violations)
    col3.metric("Rules with violations", business_rules.rules_with_violations)

    rows = [
        {
            "Rule": rule.rule_name,
            "Column": rule.column_name,
            "Type": rule.rule_type,
            "Status": rule.status,
            "Violations": rule.violation_count,
            "Violation %": rule.violation_percentage,
            "Severity": rule.severity,
        }
        for rule in business_rules.results
    ]

    st.dataframe(rows, use_container_width=True)


def display_action_log(result) -> None:
    st.subheader("Action log")

    rows = [
        {
            "Step": log.step_name,
            "Status": log.status,
            "Message": log.message,
        }
        for log in result.action_log
    ]

    st.dataframe(rows, use_container_width=True)

def display_bi_readiness_score(result) -> None:
    st.subheader("BI-readiness score")

    if result.bi_readiness_score is None:
        st.info("BI-readiness score was not computed.")
        return

    score = result.bi_readiness_score

    col1, col2 = st.columns(2)
    col1.metric("BI-readiness score", f"{score.overall_score}/{score.max_score}")
    col2.metric("Rating", score.rating)

    st.write(score.summary)

    rows = [
        {
            "Component": component.component_name,
            "Score": f"{component.score}/{component.max_score}",
            "Status": component.status,
            "Explanation": component.explanation,
        }
        for component in score.breakdown
    ]

    st.dataframe(rows, use_container_width=True)

    st.write("Recommendations:")
    for recommendation in score.recommendations:
        st.write(f"- {recommendation}")

def main() -> None:
    st.title("🧪 Agentic Data Quality Copilot")

    st.write(
        "Upload a CSV dataset, configure optional checks, and generate a data "
        "quality report using deterministic Python tools."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to start.")
        return

    try:
        dataframe = load_uploaded_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read CSV file: {error}")
        return

    dataset_name = uploaded_file.name

    st.success(f"Loaded `{dataset_name}` successfully.")

    st.subheader("Dataset preview")
    st.write(f"Rows: `{len(dataframe)}` | Columns: `{len(dataframe.columns)}`")
    st.dataframe(dataframe.head(20), use_container_width=True)

    st.divider()

    st.header("Check configuration")

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

    st.divider()

    run_checks = st.button("Run data quality checks", type="primary")

    if not run_checks:
        return

    config = DataQualityRunConfig(
        dataset_name=dataset_name,
        duplicate_key_columns=duplicate_key_columns or None,
        expected_schema=expected_schema or None,
        business_rules=business_rules,
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

    st.header("Data quality results")

    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Rows", result.schema_profile.row_count)
    overview_col2.metric("Columns", result.schema_profile.column_count)
    overview_col3.metric("Missing values", result.missing_values.total_missing_values)

    tab_schema, tab_missing, tab_duplicates, tab_types, tab_rules, tab_score, tab_log, tab_report = st.tabs(
        [
            "Schema",
            "Missing values",
            "Duplicates",
            "Type validation",
            "Business rules",
            "BI-readiness score",
            "Action log",
            "Markdown report",
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

    with tab_report:
        st.markdown(report)

        st.download_button(
            label="Download Markdown report",
            data=report,
            file_name=f"{dataset_name}_quality_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()