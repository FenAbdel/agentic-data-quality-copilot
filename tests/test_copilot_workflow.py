from pathlib import Path

from dq_copilot.agent.copilot import run_copilot_analysis
from dq_copilot.loaders.csv_loader import load_csv


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_copilot_analysis_returns_plan_result_and_report():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
        user_goal="Check if this customer dataset is ready for BI reporting.",
    )

    assert result.user_goal == "Check if this customer dataset is ready for BI reporting."
    assert result.dataset_name == "customers.csv"

    assert result.planning_result is not None
    assert result.data_quality_result is not None
    assert result.markdown_report

    assert result.planning_result.dataset_name == "customers.csv"
    assert result.data_quality_result.dataset_name == "customers.csv"

    assert "# Data Quality Report — customers.csv" in result.markdown_report


def test_copilot_analysis_creates_deterministic_plan():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    assert result.planning_result.planner_type == "deterministic_rules"
    assert len(result.planning_result.plan.steps) == 8

    tool_sequence = [
        step.tool_name for step in result.planning_result.plan.steps
    ]

    assert tool_sequence == [
        "profile_schema",
        "check_missing_values",
        "check_duplicates",
        "check_type_validation",
        "check_business_rules",
        "run_duckdb_analysis",
        "compute_bi_readiness_score",
        "generate_markdown_report",
    ]


def test_copilot_analysis_executes_planned_configuration():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    run_config = result.planning_result.run_config
    dq_result = result.data_quality_result

    assert run_config.duplicate_key_columns == ["customer_id"]
    assert run_config.expected_schema is not None
    assert run_config.business_rules
    assert run_config.sql_queries

    assert dq_result.duplicates.scope == "key_columns"
    assert dq_result.type_validation is not None
    assert dq_result.business_rules is not None
    assert dq_result.sql_analysis is not None
    assert dq_result.bi_readiness_score is not None


def test_copilot_analysis_execution_summary_is_human_readable():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    assert "Copilot analyzed dataset" in result.execution_summary
    assert "customers.csv" in result.execution_summary
    assert "BI-readiness score" in result.execution_summary