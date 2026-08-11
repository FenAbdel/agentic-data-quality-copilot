from pathlib import Path

import pandas as pd

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.models import BusinessRuleConfig, DataQualityRunConfig


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_check_runner_runs_core_checks():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
    )

    result = run_data_quality_checks(
        dataframe=dataframe,
        config=config,
    )

    assert result.dataset_name == "customers.csv"

    assert result.schema_profile.row_count == 5
    assert result.schema_profile.column_count == 6

    assert result.missing_values.total_missing_values == 2
    assert result.duplicates.status == "passed"

    assert result.type_validation is None
    assert result.business_rules is None

    assert len(result.action_log) == 5
    assert result.action_log[0].step_name == "schema_profile"
    assert result.action_log[0].status == "completed"

    assert result.action_log[3].step_name == "type_validation"
    assert result.action_log[3].status == "skipped"

    assert result.action_log[4].step_name == "business_rules"
    assert result.action_log[4].status == "skipped"


def test_check_runner_runs_type_validation_when_schema_is_provided():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
        expected_schema={
            "customer_id": "integer",
            "customer_name": "string",
            "email": "string",
            "age": "integer",
            "signup_date": "date",
            "country": "string",
        },
    )

    result = run_data_quality_checks(
        dataframe=dataframe,
        config=config,
    )

    assert result.type_validation is not None
    assert result.type_validation.status == "passed"
    assert result.type_validation.columns_checked == 6
    assert result.type_validation.total_invalid_values == 0

    type_validation_log = result.action_log[3]

    assert type_validation_log.step_name == "type_validation"
    assert type_validation_log.status == "completed"


def test_check_runner_runs_business_rules_when_rules_are_provided():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
        business_rules=[
            BusinessRuleConfig(
                rule_name="customer_id_required",
                column_name="customer_id",
                rule_type="not_null",
            ),
            BusinessRuleConfig(
                rule_name="valid_age_range",
                column_name="age",
                rule_type="range",
                min_value=0,
                max_value=120,
            ),
            BusinessRuleConfig(
                rule_name="valid_country",
                column_name="country",
                rule_type="allowed_values",
                allowed_values=["France", "Morocco", "Spain"],
            ),
        ],
    )

    result = run_data_quality_checks(
        dataframe=dataframe,
        config=config,
    )

    assert result.business_rules is not None
    assert result.business_rules.rules_checked == 3
    assert result.business_rules.total_violations == 0

    business_rules_log = result.action_log[4]

    assert business_rules_log.step_name == "business_rules"
    assert business_rules_log.status == "completed"


def test_check_runner_supports_duplicate_key_columns():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "name": ["Amina", "Amina Updated", "Karim"],
            "country": ["France", "France", "Morocco"],
        }
    )

    config = DataQualityRunConfig(
        dataset_name="duplicated_customers.csv",
        duplicate_key_columns=["customer_id"],
    )

    result = run_data_quality_checks(
        dataframe=dataframe,
        config=config,
    )

    assert result.duplicates.status == "failed"
    assert result.duplicates.scope == "key_columns"
    assert result.duplicates.key_columns == ["customer_id"]
    assert result.duplicates.duplicate_row_count == 2