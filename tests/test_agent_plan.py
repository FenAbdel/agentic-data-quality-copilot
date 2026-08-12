from dq_copilot.agent.plan_templates import create_default_bi_readiness_plan
from dq_copilot.agent.planning_models import AgentPlan, AgentPlanStep


def test_agent_plan_models_can_be_created():
    step = AgentPlanStep(
        step_id=1,
        tool_name="profile_schema",
        objective="Inspect dataset schema.",
        rationale="The schema is needed before choosing checks.",
        expected_output="Schema profile.",
        input_requirements=["dataset"],
    )

    plan = AgentPlan(
        user_goal="Check if this dataset is ready for BI.",
        dataset_name="customers.csv",
        planning_summary="Basic BI-readiness plan.",
        steps=[step],
    )

    assert plan.user_goal == "Check if this dataset is ready for BI."
    assert plan.dataset_name == "customers.csv"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "profile_schema"


def test_default_bi_readiness_plan_contains_expected_workflow():
    plan = create_default_bi_readiness_plan(
        dataset_name="customers.csv",
        user_goal="Assess BI readiness.",
    )

    tool_sequence = [step.tool_name for step in plan.steps]

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


def test_default_bi_readiness_plan_marks_advanced_steps_as_optional():
    plan = create_default_bi_readiness_plan(dataset_name="customers.csv")

    steps_by_tool = {step.tool_name: step for step in plan.steps}

    assert steps_by_tool["check_type_validation"].is_optional is True
    assert steps_by_tool["check_business_rules"].is_optional is True
    assert steps_by_tool["run_duckdb_analysis"].is_optional is True

    assert steps_by_tool["check_missing_values"].is_optional is False
    assert steps_by_tool["check_duplicates"].is_optional is False


def test_default_bi_readiness_plan_contains_dependencies():
    plan = create_default_bi_readiness_plan(dataset_name="customers.csv")

    steps_by_tool = {step.tool_name: step for step in plan.steps}

    assert steps_by_tool["check_missing_values"].depends_on == [1]
    assert steps_by_tool["check_duplicates"].depends_on == [1]
    assert steps_by_tool["compute_bi_readiness_score"].depends_on == [2, 3, 4, 5, 6]
    assert steps_by_tool["generate_markdown_report"].depends_on == [7]


def test_default_bi_readiness_plan_documents_assumptions_and_risks():
    plan = create_default_bi_readiness_plan(dataset_name="customers.csv")

    assert len(plan.assumptions) > 0
    assert len(plan.risks) > 0
    assert "Deterministic tools must calculate all metrics." in plan.assumptions