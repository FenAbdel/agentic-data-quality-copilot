import json
from pathlib import Path

from dq_copilot.agent.llm_planner import LLMPlanner
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.profiling.schema_profiler import profile_schema


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponsesClient:
    def __init__(self, output_payload: dict) -> None:
        self.output_payload = output_payload
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return FakeResponse(json.dumps(self.output_payload))


class FakeOpenAIClient:
    def __init__(self, output_payload: dict) -> None:
        self.responses = FakeResponsesClient(output_payload)


def test_llm_planner_creates_valid_planning_result_with_fake_client():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    fake_payload = {
        "planning_summary": "LLM generated a BI-readiness validation plan.",
        "duplicate_key_columns": ["customer_id"],
        "expected_schema": {
            "customer_id": "integer",
            "customer_name": "string",
            "email": "string",
            "age": "integer",
            "signup_date": "date",
            "country": "string",
        },
        "business_rules": [
            {
                "rule_name": "customer_id_required",
                "column_name": "customer_id",
                "rule_type": "not_null",
                "min_value": None,
                "max_value": None,
                "allowed_values": None,
                "allow_null": False,
                "description": "Customer identifier should be present.",
            },
            {
                "rule_name": "age_valid_range",
                "column_name": "age",
                "rule_type": "range",
                "min_value": 0,
                "max_value": 120,
                "allowed_values": None,
                "allow_null": True,
                "description": "Age should be realistic.",
            },
        ],
        "sql_queries": [
            {
                "query_name": "total_rows",
                "sql": "SELECT COUNT(*) AS row_count FROM dataset",
                "description": "Count rows.",
            },
            {
                "query_name": "country_distribution",
                "sql": (
                    "SELECT country, COUNT(*) AS row_count "
                    "FROM dataset "
                    "GROUP BY country "
                    "ORDER BY row_count DESC"
                ),
                "description": "Analyze country distribution.",
            },
        ],
        "planning_notes": [
            "customer_id appears to be a likely key column.",
            "signup_date should be validated as a date.",
        ],
        "assumptions": [
            "The dataset is intended for BI reporting.",
        ],
        "risks": [
            "Business rules are inferred and should be validated by a stakeholder.",
        ],
    }

    fake_client = FakeOpenAIClient(fake_payload)

    planner = LLMPlanner(
        client=fake_client,
        model="fake-model",
        use_fallback=False,
    )

    planning_result = planner.create_plan(
        schema_profile=schema_profile,
        user_goal="Assess BI readiness.",
    )

    assert planning_result.planner_type == "llm"
    assert planning_result.dataset_name == "customers.csv"
    assert planning_result.run_config.duplicate_key_columns == ["customer_id"]
    assert planning_result.run_config.expected_schema["signup_date"] == "date"
    assert len(planning_result.run_config.business_rules) == 2
    assert len(planning_result.run_config.sql_queries) == 2
    assert "LLM planner generated" in planning_result.planning_notes[0]


def test_llm_planner_sends_json_mode_request_to_client():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    schema_profile = profile_schema(dataframe, dataset_name="customers.csv")

    fake_payload = {
        "planning_summary": "Plan.",
        "duplicate_key_columns": ["customer_id"],
        "expected_schema": {"customer_id": "integer"},
        "business_rules": [],
        "sql_queries": [],
        "planning_notes": [],
        "assumptions": [],
        "risks": [],
    }

    fake_client = FakeOpenAIClient(fake_payload)

    planner = LLMPlanner(
        client=fake_client,
        model="fake-model",
        use_fallback=False,
    )

    planner.create_plan(
        schema_profile=schema_profile,
        user_goal="Assess BI readiness.",
    )

    request = fake_client.responses.last_request

    assert request["model"] == "fake-model"
    assert request["text"] == {"format": {"type": "json_object"}}
    assert "instructions" in request
    assert "input" in request