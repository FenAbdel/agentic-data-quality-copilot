from typing import Literal

from pydantic import BaseModel, Field

from dq_copilot.models import DataQualityRunConfig


AgentToolName = Literal[
    "profile_schema",
    "check_missing_values",
    "check_duplicates",
    "check_type_validation",
    "check_business_rules",
    "run_duckdb_analysis",
    "compute_bi_readiness_score",
    "generate_markdown_report",
]


class AgentPlanStep(BaseModel):
    step_id: int
    tool_name: AgentToolName
    objective: str
    rationale: str
    expected_output: str
    input_requirements: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)
    is_optional: bool = False
    execution_condition: str | None = None


class AgentPlan(BaseModel):
    user_goal: str
    dataset_name: str
    planning_summary: str
    steps: list[AgentPlanStep]
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class AgentPlanningResult(BaseModel):
    planner_type: Literal["deterministic_rules"] = "deterministic_rules"
    user_goal: str
    dataset_name: str
    plan: AgentPlan
    run_config: DataQualityRunConfig
    planning_notes: list[str] = Field(default_factory=list)