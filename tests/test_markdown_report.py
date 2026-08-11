from pathlib import Path

from dq_copilot.agent.check_runner import run_data_quality_checks
from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.models import BusinessRuleConfig, DataQualityRunConfig
from dq_copilot.reporting.markdown_report import (
    generate_markdown_report,
    save_markdown_report,
)


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_generate_markdown_report_contains_main_sections():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
        duplicate_key_columns=["customer_id"],
        expected_schema={
            "customer_id": "integer",
            "customer_name": "string",
            "email": "string",
            "age": "integer",
            "signup_date": "date",
            "country": "string",
        },
        business_rules=[
            BusinessRuleConfig(
                rule_name="customer_id_required",
                column_name="customer_id",
                rule_type="not_null",
            ),
            BusinessRuleConfig(
                rule_name="valid_country",
                column_name="country",
                rule_type="allowed_values",
                allowed_values=["France", "Morocco", "Spain"],
            ),
        ],
    )

    result = run_data_quality_checks(dataframe, config)
    report = generate_markdown_report(result)
    assert "## DuckDB SQL analysis" in report
    assert "# Data Quality Report — customers.csv" in report
    assert "## Dataset overview" in report
    assert "## Schema summary" in report
    assert "## Missing value analysis" in report
    assert "## Duplicate detection" in report
    assert "## Type validation" in report
    assert "## Business-rule checks" in report
    assert "## BI-readiness score" in report
    assert "## Action log" in report
    assert "## Recommendations" in report


def test_generate_markdown_report_contains_key_metrics():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
        duplicate_key_columns=["customer_id"],
    )

    result = run_data_quality_checks(dataframe, config)
    report = generate_markdown_report(result)
    assert "DuckDB SQL analysis was skipped" in report
    assert "**Rows:** 5" in report
    assert "**Columns:** 6" in report
    assert "**Total missing values:** 2" in report
    assert "**Duplicated rows:** 0" in report
    assert "Type validation was skipped" in report
    assert "Business-rule checks were skipped" in report
    assert "**Overall score:**" in report
    assert "**Rating:**" in report


def test_generate_markdown_report_includes_recommendations_for_issues():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    config = DataQualityRunConfig(
        dataset_name="customers.csv",
    )

    result = run_data_quality_checks(dataframe, config)
    report = generate_markdown_report(result)

    assert "Review columns with missing values" in report


def test_save_markdown_report_writes_file(tmp_path):
    report = "# Test Report\n\nThis is a test."

    output_path = tmp_path / "reports" / "test_report.md"

    saved_path = save_markdown_report(report, output_path)

    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == report