from dq_copilot.agent.planning_models import AgentPlan, AgentPlanStep


def create_default_bi_readiness_plan(
    dataset_name: str,
    user_goal: str = "Assess whether this dataset is ready for BI reporting.",
) -> AgentPlan:
    """
    Create a default data quality plan for BI-readiness assessment.

    This is not an LLM planner yet.
    It is a deterministic planning template that defines the standard workflow.

    Later, the LLM planner will be expected to produce a similar structured plan.
    """
    steps = [
        AgentPlanStep(
            step_id=1,
            tool_name="profile_schema",
            objective="Inspect the dataset structure.",
            rationale=(
                "Schema inspection is required before selecting relevant data quality checks."
            ),
            expected_output="Dataset shape, column names, data types, missing counts, and sample values.",
            input_requirements=["uploaded dataset"],
        ),
        AgentPlanStep(
            step_id=2,
            tool_name="check_missing_values",
            objective="Detect missing values across all columns.",
            rationale=(
                "Missing values can reduce trust in BI metrics and break downstream transformations."
            ),
            expected_output="Missing value counts, percentages, affected columns, and severity.",
            input_requirements=["dataset"],
            depends_on=[1],
        ),
        AgentPlanStep(
            step_id=3,
            tool_name="check_duplicates",
            objective="Detect duplicated rows or duplicated business keys.",
            rationale=(
                "Duplicates can inflate KPIs, create wrong joins, and corrupt dashboard results."
            ),
            expected_output="Duplicate row count, duplicate percentage, duplicate groups, and severity.",
            input_requirements=["dataset", "optional duplicate key columns"],
            depends_on=[1],
        ),
        AgentPlanStep(
            step_id=4,
            tool_name="check_type_validation",
            objective="Validate values against expected column types.",
            rationale=(
                "Type validation checks whether values match the expected data contract."
            ),
            expected_output="Invalid values, invalid percentages, and example rows.",
            input_requirements=["dataset", "expected schema"],
            depends_on=[1],
            is_optional=True,
            execution_condition="Run only when an expected schema is provided or inferred.",
        ),
        AgentPlanStep(
            step_id=5,
            tool_name="check_business_rules",
            objective="Validate configured business rules.",
            rationale=(
                "Business rules check whether technically valid values make business sense."
            ),
            expected_output="Rule violations, violation percentages, and example rows.",
            input_requirements=["dataset", "business rule configuration"],
            depends_on=[1],
            is_optional=True,
            execution_condition="Run only when business rules are configured or inferred.",
        ),
        AgentPlanStep(
            step_id=6,
            tool_name="run_duckdb_analysis",
            objective="Run optional SQL-based analytical checks.",
            rationale=(
                "SQL analysis supports BI-style exploration such as grouped counts and aggregations."
            ),
            expected_output="SQL query results or SQL validation errors.",
            input_requirements=["dataset", "read-only SQL queries"],
            depends_on=[1],
            is_optional=True,
            execution_condition="Run only when SQL analysis queries are configured or generated safely.",
        ),
        AgentPlanStep(
            step_id=7,
            tool_name="compute_bi_readiness_score",
            objective="Compute an explainable BI-readiness score.",
            rationale=(
                "A summary score helps users quickly understand whether the dataset is reliable enough for BI usage."
            ),
            expected_output="Overall score, rating, component breakdown, and recommendations.",
            input_requirements=["completed quality check results"],
            depends_on=[2, 3, 4, 5, 6],
        ),
        AgentPlanStep(
            step_id=8,
            tool_name="generate_markdown_report",
            objective="Generate a clear data quality report.",
            rationale=(
                "The final report communicates findings, risks, action logs, and recommendations to users."
            ),
            expected_output="Markdown report containing all check results and recommendations.",
            input_requirements=["data quality run result"],
            depends_on=[7],
        ),
    ]

    return AgentPlan(
        user_goal=user_goal,
        dataset_name=dataset_name,
        planning_summary=(
            "Create a BI-readiness data quality plan by inspecting the dataset, "
            "running deterministic checks, computing a score, and generating a report."
        ),
        steps=steps,
        assumptions=[
            "The uploaded dataset is tabular.",
            "The dataset is intended for BI, reporting, or downstream analytical use.",
            "Deterministic tools must calculate all metrics.",
        ],
        risks=[
            "Type validation is limited if no expected schema is provided.",
            "Business validity is limited if no business rules are configured.",
            "SQL analysis must remain read-only and controlled.",
        ],
    )