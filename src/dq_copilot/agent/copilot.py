from __future__ import annotations

import pandas as pd

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.agent.deterministic_planner import create_deterministic_planning_result
from dq_copilot.agent.planning_models import CopilotAnalysisResult
from dq_copilot.profiling.schema_profiler import profile_schema
from dq_copilot.reporting.markdown_report import generate_markdown_report


DEFAULT_USER_GOAL = "Assess whether this dataset is ready for BI reporting."


def _build_execution_summary(result: CopilotAnalysisResult) -> str:
    """
    Build a short human-readable execution summary.

    This is not an LLM summary. It is deterministic and based on real results.
    """
    dq_result = result.data_quality_result
    score = dq_result.bi_readiness_score

    if score is None:
        score_text = "No BI-readiness score was computed."
    else:
        score_text = (
            f"BI-readiness score is {score.overall_score}/{score.max_score} "
            f"with rating '{score.rating}'."
        )

    return (
        f"Copilot analyzed dataset '{result.dataset_name}' for goal: "
        f"'{result.user_goal}'. "
        f"It planned {len(result.planning_result.plan.steps)} steps, "
        f"executed deterministic data quality checks, and generated a Markdown report. "
        f"{score_text}"
    )


def run_copilot_analysis(
    dataframe: pd.DataFrame,
    dataset_name: str = "dataset",
    user_goal: str = DEFAULT_USER_GOAL,
) -> CopilotAnalysisResult:
    """
    Run the full deterministic copilot workflow.

    Workflow:
    1. Profile the dataset schema
    2. Create a deterministic agent plan
    3. Build an executable run configuration
    4. Execute deterministic data quality checks
    5. Generate a Markdown report
    6. Return plan, results, report, and summary

    This is the first complete agentic workflow, but it does not use an LLM yet.
    """
    schema_profile = profile_schema(
        dataframe=dataframe,
        dataset_name=dataset_name,
    )

    planning_result = create_deterministic_planning_result(
        schema_profile=schema_profile,
        user_goal=user_goal,
    )

    data_quality_result = run_data_quality_checks(
        dataframe=dataframe,
        config=planning_result.run_config,
    )

    markdown_report = generate_markdown_report(data_quality_result)

    copilot_result = CopilotAnalysisResult(
        user_goal=user_goal,
        dataset_name=dataset_name,
        planning_result=planning_result,
        data_quality_result=data_quality_result,
        markdown_report=markdown_report,
        execution_summary="",
    )

    copilot_result.execution_summary = _build_execution_summary(copilot_result)

    return copilot_result