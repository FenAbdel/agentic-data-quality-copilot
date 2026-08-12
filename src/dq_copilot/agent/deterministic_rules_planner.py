from dq_copilot.agent.deterministic_planner import (
    create_deterministic_planning_result,
)
from dq_copilot.agent.planning_models import AgentPlanningResult
from dq_copilot.models import DatasetSchemaProfile


class DeterministicRulesPlanner:
    """
    Rule-based planner.

    This planner uses deterministic heuristics to infer:
    - likely duplicate key columns
    - expected schema
    - simple business rules
    - useful DuckDB SQL analysis queries

    It does not use an LLM.
    """

    planner_type = "deterministic_rules"

    def create_plan(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> AgentPlanningResult:
        return create_deterministic_planning_result(
            schema_profile=schema_profile,
            user_goal=user_goal,
        )