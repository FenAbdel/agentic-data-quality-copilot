from pathlib import Path

from dq_copilot.agent.copilot import run_copilot_analysis
from dq_copilot.agent.observer import build_copilot_observations
from dq_copilot.loaders.csv_loader import load_csv


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_observer_builds_observations_from_copilot_result():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    copilot_result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    observations = copilot_result.observations

    assert observations
    assert any(observation.source == "missing_values" for observation in observations)
    assert any(observation.source == "duplicates" for observation in observations)
    assert any(observation.source == "bi_readiness_score" for observation in observations)
    assert any(observation.source == "result_verification" for observation in observations)


def test_observer_can_be_called_directly():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    copilot_result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
    )

    observations = build_copilot_observations(
        result=copilot_result.data_quality_result,
        verification_result=copilot_result.verification_result,
    )

    assert observations
    assert all(observation.message for observation in observations)