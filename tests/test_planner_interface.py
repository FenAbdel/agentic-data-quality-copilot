
from pathlib import Path

import pytest

from dq_copilot.agent.copilot import run_copilot_analysis
from dq_copilot.agent.deterministic_rules_planner import DeterministicRulesPlanner
from dq_copilot.agent.llm_planner import LLMPlanner, LLMPlannerNotConfiguredError
from dq_copilot.agent.planner_factory import create_planner
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.profiling.schema_profiler import profile_schema
from dq_copilot.agent.llm_planner import LLMPlanner, LLMPlannerNotConfiguredError

SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_deterministic_rules_planner_creates_planning_result():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planner = DeterministicRulesPlanner()

    planning_result = planner.create_plan(
        schema_profile=schema_profile,
        user_goal="Check BI readiness.",
    )

    assert planning_result.planner_type == "deterministic_rules"
    assert planning_result.dataset_name == "customers.csv"
    assert planning_result.run_config.dataset_name == "customers.csv"
    assert len(planning_result.plan.steps) == 8


def test_planner_factory_returns_deterministic_planner():
    planner = create_planner("deterministic_rules")

    assert planner.planner_type == "deterministic_rules"
    assert isinstance(planner, DeterministicRulesPlanner)


def test_planner_factory_returns_llm_placeholder():
    planner = create_planner("llm")

    assert planner.planner_type == "llm"
    assert isinstance(planner, LLMPlanner)


def test_llm_planner_placeholder_raises_clear_error(monkeypatch):
    monkeypatch.setattr("dq_copilot.agent.llm_planner.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planner = LLMPlanner()

    with pytest.raises(LLMPlannerNotConfiguredError):
        planner.create_plan(
            schema_profile=schema_profile,
            user_goal="Check BI readiness.",
        )


def test_copilot_accepts_planner_interface():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    planner = DeterministicRulesPlanner()

    result = run_copilot_analysis(
        dataframe=dataframe,
        dataset_name="customers.csv",
        user_goal="Check BI readiness.",
        planner=planner,
    )

    assert result.planning_result.planner_type == "deterministic_rules"
    assert result.data_quality_result.dataset_name == "customers.csv"
    assert result.markdown_report

def test_llm_planner_without_api_key_falls_back_to_deterministic_planner(monkeypatch):
    monkeypatch.setattr("dq_copilot.agent.llm_planner.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planner = LLMPlanner(use_fallback=True)

    planning_result = planner.create_plan(
        schema_profile=schema_profile,
        user_goal="Check BI readiness.",
    )

    assert planning_result.planner_type == "deterministic_rules"
    assert "Used deterministic fallback" in planning_result.planning_notes[0]


def test_llm_planner_without_api_key_can_raise_when_fallback_is_disabled(monkeypatch):
    monkeypatch.setattr("dq_copilot.agent.llm_planner.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    planner = LLMPlanner(use_fallback=False)

    with pytest.raises(LLMPlannerNotConfiguredError):
        planner.create_plan(
            schema_profile=schema_profile,
            user_goal="Check BI readiness.",
        )