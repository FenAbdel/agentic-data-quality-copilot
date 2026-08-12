from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dq_copilot.agent.deterministic_planner import create_deterministic_run_config
from dq_copilot.agent.deterministic_rules_planner import DeterministicRulesPlanner
from dq_copilot.agent.plan_templates import create_default_bi_readiness_plan
from dq_copilot.agent.planning_models import AgentPlanningResult
from dq_copilot.models import (
    BusinessRuleConfig,
    DataQualityRunConfig,
    DatasetSchemaProfile,
    ExpectedDataType,
    SQLAnalysisQuery,
)
from dq_copilot.sql.duckdb_analyzer import validate_read_only_sql


class LLMPlannerError(RuntimeError):
    """
    Base error for LLM planner failures.
    """


class LLMPlannerNotConfiguredError(LLMPlannerError):
    """
    Raised when the LLM planner is requested without API configuration.
    """


class LLMPlannerOutput(BaseModel):
    """
    Strict structure expected from the LLM.

    The LLM does not return metrics.
    It only returns planning/configuration suggestions.
    """

    planning_summary: str
    duplicate_key_columns: list[str] | None = None
    expected_schema: dict[str, ExpectedDataType] = Field(default_factory=dict)
    business_rules: list[BusinessRuleConfig] = Field(default_factory=list)
    sql_queries: list[SQLAnalysisQuery] = Field(default_factory=list)
    planning_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class LLMPlanner:
    """
    LLM-based planner.

    The LLM receives schema metadata and a user goal.
    It proposes a structured run configuration.

    Important:
    - The LLM does not calculate quality metrics.
    - The LLM does not execute SQL.
    - The LLM output is validated with Pydantic.
    - If the LLM fails, the planner can fall back to deterministic rules.
    """

    planner_type = "llm"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        use_fallback: bool = False,
    ) -> None:
        load_dotenv()

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client
        self.use_fallback = use_fallback
        self.fallback_planner = DeterministicRulesPlanner()

    def create_plan(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> AgentPlanningResult:
        try:
            llm_output = self._call_llm(
                schema_profile=schema_profile,
                user_goal=user_goal,
            )

            self._validate_llm_output(
                llm_output=llm_output,
                schema_profile=schema_profile,
            )

            return self._build_planning_result(
                llm_output=llm_output,
                schema_profile=schema_profile,
                user_goal=user_goal,
            )

        except Exception as error:
            if self.use_fallback:
                return self._fallback_to_deterministic_planner(
                    schema_profile=schema_profile,
                    user_goal=user_goal,
                    error=error,
                )

            if isinstance(error, LLMPlannerError):
                raise

            raise LLMPlannerError(f"LLM planner failed: {error}") from error

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client

        if not self.api_key:
            raise LLMPlannerNotConfiguredError(
                "OPENAI_API_KEY is not configured. "
                "Add it to your .env file or use DeterministicRulesPlanner."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)
        return self.client

    def _call_llm(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> LLMPlannerOutput:
        client = self._get_client()

        prompt_payload = self._build_prompt_payload(
            schema_profile=schema_profile,
            user_goal=user_goal,
        )

        response = client.responses.create(
            model=self.model,
            instructions=self._build_system_instructions(),
            input=(
                "Respond with a single JSON object only.\n\n"
                + json.dumps(prompt_payload, indent=2)
            ),
            text={"format": {"type": "json_object"}},
        )

        raw_text = response.output_text

        try:
            return LLMPlannerOutput.model_validate_json(raw_text)
        except Exception as error:
            raise LLMPlannerError(
                f"LLM returned invalid planning JSON: {raw_text}"
            ) from error

    def _build_system_instructions(self) -> str:
        return """
You are the planning component of an Agentic Data Quality Copilot.

Your job:
- Read schema metadata and a user goal.
- Propose a data quality run configuration.
- Return JSON only.
- Do not calculate data quality metrics.
- Do not invent row counts, missing counts, duplicate counts, scores, or results.
- Do not claim that checks passed or failed.
- Only propose what deterministic tools should check.

Available expected types:
- integer
- float
- numeric
- date
- string
- boolean

Available business rule types:
- not_null
- range
- positive
- non_negative
- allowed_values
- not_future_date

SQL rules:
- SQL queries must be read-only.
- Only SELECT or WITH queries are allowed.
- The table name must be dataset.
- Do not use DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, COPY, INSTALL, LOAD, ATTACH, PRAGMA, or SET.

Privacy rule:
- You receive schema metadata only.
- Do not ask for full raw data.
- Do not require sensitive values.

Return JSON matching this shape:
{
  "planning_summary": "...",
  "duplicate_key_columns": ["..."] or null,
  "expected_schema": {
    "column_name": "integer|float|numeric|date|string|boolean"
  },
  "business_rules": [
    {
      "rule_name": "...",
      "column_name": "...",
      "rule_type": "not_null|range|positive|non_negative|allowed_values|not_future_date",
      "min_value": null,
      "max_value": null,
      "allowed_values": null,
      "allow_null": true,
      "description": "..."
    }
  ],
  "sql_queries": [
    {
      "query_name": "...",
      "sql": "SELECT ... FROM dataset",
      "description": "..."
    }
  ],
  "planning_notes": ["..."],
  "assumptions": ["..."],
  "risks": ["..."]
}
""".strip()

    def _build_prompt_payload(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> dict[str, Any]:
        return {
            "user_goal": user_goal,
            "dataset_name": schema_profile.dataset_name,
            "row_count": schema_profile.row_count,
            "column_count": schema_profile.column_count,
            "columns": [
                {
                    "name": column.name,
                    "pandas_dtype": column.pandas_dtype,
                    "non_null_count": column.non_null_count,
                    "null_count": column.null_count,
                    "null_percentage": column.null_percentage,
                    "sample_values_redacted": True,
                }
                for column in schema_profile.columns
            ],
            "important_instruction": (
                "Create a planning configuration only. "
                "Do not calculate check results. "
                "Use only column names and schema metadata."
            ),
        }

    def _validate_llm_output(
        self,
        llm_output: LLMPlannerOutput,
        schema_profile: DatasetSchemaProfile,
    ) -> None:
        known_columns = {column.name for column in schema_profile.columns}

        if llm_output.duplicate_key_columns:
            unknown_key_columns = [
                column
                for column in llm_output.duplicate_key_columns
                if column not in known_columns
            ]
            if unknown_key_columns:
                raise LLMPlannerError(
                    f"LLM suggested unknown duplicate key columns: {unknown_key_columns}"
                )

        unknown_expected_schema_columns = [
            column
            for column in llm_output.expected_schema
            if column not in known_columns
        ]
        if unknown_expected_schema_columns:
            raise LLMPlannerError(
                "LLM suggested expected schema for unknown columns: "
                f"{unknown_expected_schema_columns}"
            )

        unknown_rule_columns = [
            rule.column_name
            for rule in llm_output.business_rules
            if rule.column_name not in known_columns
        ]
        if unknown_rule_columns:
            raise LLMPlannerError(
                f"LLM suggested business rules for unknown columns: {unknown_rule_columns}"
            )

        for query in llm_output.sql_queries:
            validate_read_only_sql(query.sql)

    def _build_planning_result(
        self,
        llm_output: LLMPlannerOutput,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
    ) -> AgentPlanningResult:
        deterministic_config = create_deterministic_run_config(schema_profile)

        expected_schema = {
            **(deterministic_config.expected_schema or {}),
            **llm_output.expected_schema,
        }

        business_rules = (
            llm_output.business_rules
            if llm_output.business_rules
            else deterministic_config.business_rules
        )

        sql_queries = (
            llm_output.sql_queries
            if llm_output.sql_queries
            else deterministic_config.sql_queries
        )

        run_config = DataQualityRunConfig(
            dataset_name=schema_profile.dataset_name,
            duplicate_key_columns=(
                llm_output.duplicate_key_columns
                or deterministic_config.duplicate_key_columns
            ),
            expected_schema=expected_schema,
            business_rules=business_rules,
            sql_queries=sql_queries,
        )

        plan = create_default_bi_readiness_plan(
            dataset_name=schema_profile.dataset_name,
            user_goal=user_goal,
        )
        plan.planning_summary = llm_output.planning_summary

        if llm_output.assumptions:
            plan.assumptions = llm_output.assumptions

        if llm_output.risks:
            plan.risks = llm_output.risks

        planning_notes = [
            "LLM planner generated the run configuration. "
            "All metrics will still be calculated by deterministic tools."
        ]
        planning_notes.extend(llm_output.planning_notes)

        return AgentPlanningResult(
            planner_type="llm",
            user_goal=user_goal,
            dataset_name=schema_profile.dataset_name,
            plan=plan,
            run_config=run_config,
            planning_notes=planning_notes,
        )

    def _fallback_to_deterministic_planner(
        self,
        schema_profile: DatasetSchemaProfile,
        user_goal: str,
        error: Exception,
    ) -> AgentPlanningResult:
        fallback_result = self.fallback_planner.create_plan(
            schema_profile=schema_profile,
            user_goal=user_goal,
        )

        fallback_result.planning_notes.insert(
            0,
            (
                "LLM planner was unavailable or returned invalid output. "
                f"Used deterministic fallback instead. Reason: {error}"
            ),
        )

        return fallback_result