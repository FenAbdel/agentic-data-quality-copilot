from pathlib import Path

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.models import BusinessRuleConfig, DataQualityRunConfig
from dq_copilot.scoring.bi_readiness import compute_bi_readiness_score


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_bi_readiness_score_is_computed_from_run_result():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
    )

    result = run_data_quality_checks(dataframe, config)

    assert result.bi_readiness_score is not None
    assert result.bi_readiness_score.max_score == 100
    assert 0 <= result.bi_readiness_score.overall_score <= 100
    assert result.bi_readiness_score.rating in {
        "excellent",
        "good",
        "needs_attention",
        "poor",
    }
    assert len(result.bi_readiness_score.breakdown) == 5


def test_bi_readiness_score_penalizes_skipped_advanced_checks():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
    )

    result = run_data_quality_checks(dataframe, config)
    score = result.bi_readiness_score

    assert score is not None
    assert score.overall_score < 75

    components = {component.component_name: component for component in score.breakdown}

    assert components["type_validation"].status == "skipped"
    assert components["business_rules"].status == "skipped"
    assert components["check_coverage"].score == 5


def test_bi_readiness_score_improves_when_advanced_checks_are_configured():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
        duplicate_key_columns=["customer_id"],
        expected_schema={
            "customer_id": "integer",
            "customer_name": "string",
            "email": "string",
            "age": "integer",
            "signup_date": "date",
            "country": "string",
        },
        business_rules=[
            BusinessRuleConfig(
                rule_name="customer_id_required",
                column_name="customer_id",
                rule_type="not_null",
            ),
            BusinessRuleConfig(
                rule_name="valid_country",
                column_name="country",
                rule_type="allowed_values",
                allowed_values=["France", "Morocco", "Spain"],
            ),
        ],
    )

    result = run_data_quality_checks(dataframe, config)
    score = result.bi_readiness_score

    assert score is not None
    assert score.overall_score >= 75
    assert score.rating in {"good", "excellent"}

    components = {component.component_name: component for component in score.breakdown}

    assert components["type_validation"].status == "strong"
    assert components["business_rules"].status == "strong"
    assert components["check_coverage"].score == 10


def test_compute_bi_readiness_score_can_be_called_directly():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
    )

    result = run_data_quality_checks(dataframe, config)

    score = compute_bi_readiness_score(result)

    assert score.overall_score == result.bi_readiness_score.overall_score
    assert score.rating == result.bi_readiness_score.rating