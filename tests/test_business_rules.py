from datetime import date

import pandas as pd
import pytest

from dq_copilot.checks.business_rules import (
    check_business_rules,
    classify_business_rule_severity,
)
from dq_copilot.models import BusinessRuleConfig


def test_business_rules_pass_when_values_are_valid():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [29, 34, 41],
            "country": ["France", "Morocco", "Spain"],
            "amount": [100.0, 250.0, 75.5],
            "signup_date": ["2024-01-15", "2024-02-20", "2024-03-05"],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="customer_id_required",
            column_name="customer_id",
            rule_type="not_null",
        ),
        BusinessRuleConfig(
            rule_name="valid_age_range",
            column_name="age",
            rule_type="range",
            min_value=0,
            max_value=120,
        ),
        BusinessRuleConfig(
            rule_name="valid_country",
            column_name="country",
            rule_type="allowed_values",
            allowed_values=["France", "Morocco", "Spain"],
        ),
        BusinessRuleConfig(
            rule_name="positive_amount",
            column_name="amount",
            rule_type="positive",
        ),
        BusinessRuleConfig(
            rule_name="signup_date_not_future",
            column_name="signup_date",
            rule_type="not_future_date",
        ),
    ]

    result = check_business_rules(
        dataframe=dataframe,
        rules=rules,
        reference_date=date(2026, 1, 1),
    )

    assert result.check_name == "business_rules"
    assert result.status == "passed"
    assert result.rules_checked == 5
    assert result.rules_with_violations == 0
    assert result.total_violations == 0


def test_business_rules_detect_range_violations():
    dataframe = pd.DataFrame(
        {
            "age": [29, -1, 121, "unknown"],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="valid_age_range",
            column_name="age",
            rule_type="range",
            min_value=0,
            max_value=120,
        )
    ]

    result = check_business_rules(dataframe, rules)

    age_result = result.results[0]

    assert result.status == "failed"
    assert age_result.status == "failed"
    assert age_result.violation_count == 3
    assert age_result.violation_percentage == 75.0
    assert age_result.severity == "high"
    assert len(age_result.violation_examples) == 3


def test_business_rules_detect_allowed_value_violations():
    dataframe = pd.DataFrame(
        {
            "country": ["France", "Morocco", "Unknown", "Spain"],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="valid_country",
            column_name="country",
            rule_type="allowed_values",
            allowed_values=["France", "Morocco", "Spain"],
        )
    ]

    result = check_business_rules(dataframe, rules)

    country_result = result.results[0]

    assert result.status == "failed"
    assert country_result.violation_count == 1
    assert country_result.violation_examples[0].value == "Unknown"


def test_business_rules_detect_positive_value_violations():
    dataframe = pd.DataFrame(
        {
            "amount": [100.0, 0.0, -50.0, "bad"],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="positive_amount",
            column_name="amount",
            rule_type="positive",
        )
    ]

    result = check_business_rules(dataframe, rules)

    amount_result = result.results[0]

    assert result.status == "failed"
    assert amount_result.violation_count == 3


def test_business_rules_detect_not_future_date_violations():
    dataframe = pd.DataFrame(
        {
            "signup_date": ["2024-01-15", "2026-02-01", "not-a-date", None],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="signup_date_not_future",
            column_name="signup_date",
            rule_type="not_future_date",
        )
    ]

    result = check_business_rules(
        dataframe=dataframe,
        rules=rules,
        reference_date=date(2026, 1, 1),
    )

    date_result = result.results[0]

    assert result.status == "failed"
    assert date_result.violation_count == 2
    assert len(date_result.violation_examples) == 2


def test_business_rules_detect_not_null_violations():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, None, 3],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="customer_id_required",
            column_name="customer_id",
            rule_type="not_null",
        )
    ]

    result = check_business_rules(dataframe, rules)

    customer_id_result = result.results[0]

    assert result.status == "failed"
    assert customer_id_result.violation_count == 1
    assert customer_id_result.violation_examples[0].value == "<missing>"


def test_business_rules_can_disallow_nulls_for_any_rule():
    dataframe = pd.DataFrame(
        {
            "age": [29, None, 41],
        }
    )

    rules = [
        BusinessRuleConfig(
            rule_name="age_required_and_valid",
            column_name="age",
            rule_type="range",
            min_value=0,
            max_value=120,
            allow_null=False,
        )
    ]

    result = check_business_rules(dataframe, rules)

    age_result = result.results[0]

    assert result.status == "failed"
    assert age_result.violation_count == 1
    assert age_result.violation_examples[0].value == "<missing>"


def test_business_rules_reject_invalid_rule_configuration():
    dataframe = pd.DataFrame(
        {
            "age": [29, 34],
            "country": ["France", "Morocco"],
        }
    )

    with pytest.raises(ValueError):
        check_business_rules(
            dataframe,
            [
                BusinessRuleConfig(
                    rule_name="unknown_column_rule",
                    column_name="unknown_column",
                    rule_type="not_null",
                )
            ],
        )

    with pytest.raises(ValueError):
        check_business_rules(
            dataframe,
            [
                BusinessRuleConfig(
                    rule_name="invalid_range_rule",
                    column_name="age",
                    rule_type="range",
                )
            ],
        )

    with pytest.raises(ValueError):
        check_business_rules(
            dataframe,
            [
                BusinessRuleConfig(
                    rule_name="invalid_allowed_values_rule",
                    column_name="country",
                    rule_type="allowed_values",
                    allowed_values=[],
                )
            ],
        )


def test_business_rule_severity_classification():
    assert classify_business_rule_severity(0.0) == "ok"
    assert classify_business_rule_severity(0.5) == "low"
    assert classify_business_rule_severity(3.0) == "medium"
    assert classify_business_rule_severity(10.0) == "high"


def test_business_rules_reject_invalid_thresholds():
    dataframe = pd.DataFrame({"age": [29, -1]})
    rules = [
        BusinessRuleConfig(
            rule_name="valid_age_range",
            column_name="age",
            rule_type="range",
            min_value=0,
            max_value=120,
        )
    ]

    with pytest.raises(ValueError):
        check_business_rules(dataframe, rules, warning_threshold=-1)

    with pytest.raises(ValueError):
        check_business_rules(dataframe, rules, warning_threshold=10, high_threshold=5)

    with pytest.raises(ValueError):
        check_business_rules(dataframe, rules, max_examples=-1)