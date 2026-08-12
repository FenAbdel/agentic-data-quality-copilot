from pathlib import Path

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.agent.deterministic_planner import (
    create_deterministic_planning_result,
    create_deterministic_run_config,
    infer_business_rules,
    infer_duplicate_key_columns,
    infer_expected_schema,
    infer_sql_queries,
)
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.profiling.schema_profiler import profile_schema


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_deterministic_planner_infers_duplicate_key_column():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    duplicate_key_columns = infer_duplicate_key_columns(schema_profile)

    assert duplicate_key_columns == ["customer_id"]


def test_deterministic_planner_infers_expected_schema():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    expected_schema = infer_expected_schema(schema_profile)

    assert expected_schema["customer_id"] == "integer"
    assert expected_schema["customer_name"] == "string"
    assert expected_schema["email"] == "string"
    assert expected_schema["age"] == "integer"
    assert expected_schema["signup_date"] == "date"
    assert expected_schema["country"] == "string"


def test_deterministic_planner_infers_business_rules():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    rules = infer_business_rules(schema_profile)
    rule_names = {rule.rule_name for rule in rules}

    assert "customer_id_required" in rule_names
    assert "age_valid_range" in rule_names
    assert "signup_date_not_future" in rule_names


def test_deterministic_planner_generates_sql_queries():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    queries = infer_sql_queries(schema_profile)
    query_names = {query.query_name for query in queries}

    assert "total_rows" in query_names
    assert "country_distribution" in query_names
    assert "age_numeric_summary" in query_names
    assert "signup_date_date_range" in query_names


def test_deterministic_planner_creates_run_config():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    config = create_deterministic_run_config(schema_profile)

    assert config.dataset_name == "customers.csv"
    assert config.duplicate_key_columns == ["customer_id"]
    assert config.expected_schema is not None
    assert config.expected_schema["signup_date"] == "date"
    assert len(config.business_rules) >= 3
    assert len(config.sql_queries) >= 1


def test_deterministic_planning_result_contains_plan_and_config():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planning_result = create_deterministic_planning_result(
        schema_profile=schema_profile,
        user_goal="Check this dataset for BI readiness.",
    )

    assert planning_result.planner_type == "deterministic_rules"
    assert planning_result.user_goal == "Check this dataset for BI readiness."
    assert planning_result.dataset_name == "customers.csv"
    assert len(planning_result.plan.steps) == 8
    assert planning_result.run_config.dataset_name == "customers.csv"
    assert len(planning_result.planning_notes) > 0


def test_deterministic_planner_config_can_be_executed_by_runner():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planning_result = create_deterministic_planning_result(schema_profile)

    result = run_data_quality_checks(
        dataframe=dataframe,
        config=planning_result.run_config,
    )

    assert result.schema_profile.row_count == 5
    assert result.duplicates.scope == "key_columns"
    assert result.type_validation is not None
    assert result.business_rules is not None
    assert result.sql_analysis is not None
    assert result.bi_readiness_score is not None