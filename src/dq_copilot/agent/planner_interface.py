from typing import Protocol

from dq_copilot.agent.planning_models import AgentPlanningResult, PlannerType
from dq_copilot.models import DatasetSchemaProfile


class AgentPlanner(Protocol):
    """
    Interface for all agent planners.

    A planner receives a schema profile and a user goal.
    It returns a structured planning result that can be executed by the runner.

    Implementations can be:
    - deterministic rules
    - LLM-based planner
    - future hybrid planner
    """

    planner_type: PlannerType

    def create_plan(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> AgentPlanningResult:
        """
        Create a structured agent planning result.

        The returned object must include:
        - human-readable plan
        - executable DataQualityRunConfig
        - planning notes
        """
        ...