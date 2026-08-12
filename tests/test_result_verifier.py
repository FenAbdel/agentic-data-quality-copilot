from pathlib import Path

from dq_copilot.agent.result_verifier import verify_data_quality_result
from dq_copilot.agent.copilot import run_copilot_analysis
from dq_copilot.loaders.csv_loader import load_csv


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_result_verifier_passes_for_valid_copilot_result():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    copilot_result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    verification = copilot_result.verification_result

    assert verification is not None
    assert verification.status == "passed"
    assert verification.checks_failed == 0
    assert verification.checks_run >= 6


def test_result_verifier_detects_modified_missing_total():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    copilot_result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    dq_result = copilot_result.data_quality_result
    dq_result.missing_values.total_missing_values = 999

    verification = verify_data_quality_result(
        dataframe=dataframe,
        result=dq_result,
    )

    assert verification.status == "failed"
    assert verification.checks_failed >= 1

    failed_checks = {
        check.check_name for check in verification.results if check.status == "failed"
    }

    assert "missing_values_consistency" in failed_checks or "missing_values_recomputation" in failed_checks


def test_result_verifier_detects_modified_bi_score():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    copilot_result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    dq_result = copilot_result.data_quality_result
    assert dq_result.bi_readiness_score is not None

    dq_result.bi_readiness_score.overall_score = 1

    verification = verify_data_quality_result(
        dataframe=dataframe,
        result=dq_result,
    )

    assert verification.status == "failed"

    failed_checks = {
        check.check_name for check in verification.results if check.status == "failed"
    }

    assert "bi_readiness_score_total" in failed_checks