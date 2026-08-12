from dq_copilot.agent.deterministic_rules_planner import DeterministicRulesPlanner
from dq_copilot.agent.llm_planner import LLMPlanner
from dq_copilot.agent.planner_interface import AgentPlanner
from dq_copilot.agent.planning_models import PlannerType


def create_planner(planner_type: PlannerType = "deterministic_rules") -> AgentPlanner:
    """
    Create a planner implementation.

    For now:
    - deterministic_rules works
    - llm returns the placeholder LLM planner

    The LLM planner will be implemented in the next steps.
    """
    if planner_type == "deterministic_rules":
        return DeterministicRulesPlanner()

    if planner_type == "llm":
        return LLMPlanner()

    raise ValueError(f"Unsupported planner type: {planner_type}")