from collections.abc import Sequence
from datetime import date

import pandas as pd

from dq_copilot.models import (
    BusinessRuleConfig,
    BusinessRuleResult,
    BusinessRulesCheckResult,
    BusinessRuleViolationExample,
)


def classify_business_rule_severity(
    violation_percentage: float,
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
) -> str:
    """
    Classify business-rule severity based on violation percentage.

    Rules:
    - 0% violations      -> ok
    - <= warning         -> low
    - <= high            -> medium
    - > high             -> high
    """
    if violation_percentage == 0:
        return "ok"

    if violation_percentage <= warning_threshold:
        return "low"

    if violation_percentage <= high_threshold:
        return "medium"

    return "high"


def _validate_rules(dataframe: pd.DataFrame, rules: Sequence[BusinessRuleConfig]) -> None:
    """
    Validate that configured business rules are usable.
    """
    missing_columns = [
        rule.column_name for rule in rules if rule.column_name not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Business rule columns not found: {sorted(set(missing_columns))}")

    invalid_range_rules = [
        rule.rule_name
        for rule in rules
        if rule.rule_type == "range"
        and rule.min_value is None
        and rule.max_value is None
    ]

    if invalid_range_rules:
        raise ValueError(
            f"Range rules must define min_value, max_value, or both: {invalid_range_rules}"
        )

    invalid_allowed_value_rules = [
        rule.rule_name
        for rule in rules
        if rule.rule_type == "allowed_values" and not rule.allowed_values
    ]

    if invalid_allowed_value_rules:
        raise ValueError(
            "Allowed-values rules must define at least one allowed value: "
            f"{invalid_allowed_value_rules}"
        )


def _stringify_value(value: object) -> str:
    """
    Convert a value to a display-safe string.
    """
    if pd.isna(value):
        return "<missing>"

    return str(value)


def _build_violation_message(rule: BusinessRuleConfig) -> str:
    """
    Build a human-readable violation message for a rule.
    """
    if rule.rule_type == "not_null":
        return "Value must not be missing."

    if rule.rule_type == "range":
        if rule.min_value is not None and rule.max_value is not None:
            return f"Value must be between {rule.min_value} and {rule.max_value}."

        if rule.min_value is not None:
            return f"Value must be greater than or equal to {rule.min_value}."

        return f"Value must be less than or equal to {rule.max_value}."

    if rule.rule_type == "positive":
        return "Value must be strictly positive."

    if rule.rule_type == "non_negative":
        return "Value must be greater than or equal to zero."

    if rule.rule_type == "allowed_values":
        return "Value must belong to the configured allowed values."

    if rule.rule_type == "not_future_date":
        return "Date must not be in the future."

    return "Value violates the configured business rule."


def _evaluate_rule(
    dataframe: pd.DataFrame,
    rule: BusinessRuleConfig,
    reference_date: date,
) -> pd.Series:
    """
    Evaluate one business rule and return a boolean mask.

    True means the row violates the rule.
    False means the row passes the rule.
    """
    series = dataframe[rule.column_name]
    violation_mask = pd.Series(False, index=series.index)

    null_mask = series.isna()

    if rule.rule_type == "not_null" or not rule.allow_null:
        violation_mask = violation_mask | null_mask

    values_to_validate = series[series.notna()]

    if values_to_validate.empty:
        return violation_mask

    if rule.rule_type == "not_null":
        return violation_mask

    if rule.rule_type == "range":
        converted = pd.to_numeric(values_to_validate, errors="coerce")

        invalid_mask = converted.isna()

        if rule.min_value is not None:
            invalid_mask = invalid_mask | (converted < rule.min_value)

        if rule.max_value is not None:
            invalid_mask = invalid_mask | (converted > rule.max_value)

        violation_mask.loc[values_to_validate.index] = invalid_mask
        return violation_mask

    if rule.rule_type == "positive":
        converted = pd.to_numeric(values_to_validate, errors="coerce")
        invalid_mask = converted.isna() | (converted <= 0)

        violation_mask.loc[values_to_validate.index] = invalid_mask
        return violation_mask

    if rule.rule_type == "non_negative":
        converted = pd.to_numeric(values_to_validate, errors="coerce")
        invalid_mask = converted.isna() | (converted < 0)

        violation_mask.loc[values_to_validate.index] = invalid_mask
        return violation_mask

    if rule.rule_type == "allowed_values":
        allowed_values = {str(value).strip() for value in rule.allowed_values or []}
        normalized_values = values_to_validate.astype(str).str.strip()

        invalid_mask = ~normalized_values.isin(allowed_values)

        violation_mask.loc[values_to_validate.index] = invalid_mask
        return violation_mask

    if rule.rule_type == "not_future_date":
        converted = pd.to_datetime(values_to_validate, errors="coerce")
        reference_timestamp = pd.Timestamp(reference_date).normalize()

        invalid_mask = converted.isna() | (converted.dt.normalize() > reference_timestamp)

        violation_mask.loc[values_to_validate.index] = invalid_mask
        return violation_mask

    raise ValueError(f"Unsupported business rule type: {rule.rule_type}")


def _build_violation_examples(
    dataframe: pd.DataFrame,
    rule: BusinessRuleConfig,
    violation_mask: pd.Series,
    max_examples: int,
) -> list[BusinessRuleViolationExample]:
    """
    Build display-safe examples of rule violations.
    """
    examples: list[BusinessRuleViolationExample] = []
    violating_values = dataframe.loc[violation_mask, rule.column_name]

    message = _build_violation_message(rule)

    for row_index, value in violating_values.head(max_examples).items():
        examples.append(
            BusinessRuleViolationExample(
                row_index=str(row_index),
                value=_stringify_value(value),
                message=message,
            )
        )

    return examples


def check_business_rules(
    dataframe: pd.DataFrame,
    rules: Sequence[BusinessRuleConfig],
    warning_threshold: float = 1.0,
    high_threshold: float = 5.0,
    max_examples: int = 10,
    reference_date: date | None = None,
) -> BusinessRulesCheckResult:
    """
    Execute configurable business rules against a DataFrame.

    Parameters
    ----------
    dataframe:
        Dataset to analyze.

    rules:
        Business rules to execute.

    warning_threshold:
        Percentage threshold above which violations become medium severity.

    high_threshold:
        Percentage threshold above which violations become high severity.

    max_examples:
        Maximum number of violation examples per rule.

    reference_date:
        Date used by date-based rules such as not_future_date.
        If not provided, today's date is used.

    Returns
    -------
    BusinessRulesCheckResult
        Structured business-rule validation result.
    """
    if warning_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be positive numbers.")

    if warning_threshold > high_threshold:
        raise ValueError("warning_threshold cannot be greater than high_threshold.")

    if max_examples < 0:
        raise ValueError("max_examples must be a positive number.")

    _validate_rules(dataframe, rules)

    if reference_date is None:
        reference_date = date.today()

    row_count = len(dataframe)
    rule_results: list[BusinessRuleResult] = []

    for rule in rules:
        violation_mask = _evaluate_rule(
            dataframe=dataframe,
            rule=rule,
            reference_date=reference_date,
        )

        violation_count = int(violation_mask.sum())

        if row_count == 0:
            violation_percentage = 0.0
        else:
            violation_percentage = round((violation_count / row_count) * 100, 2)

        severity = classify_business_rule_severity(
            violation_percentage=violation_percentage,
            warning_threshold=warning_threshold,
            high_threshold=high_threshold,
        )

        status = "passed" if violation_count == 0 else "failed"

        rule_results.append(
            BusinessRuleResult(
                rule_name=rule.rule_name,
                column_name=rule.column_name,
                rule_type=rule.rule_type,
                status=status,
                violation_count=violation_count,
                violation_percentage=violation_percentage,
                severity=severity,
                violation_examples=_build_violation_examples(
                    dataframe=dataframe,
                    rule=rule,
                    violation_mask=violation_mask,
                    max_examples=max_examples,
                ),
            )
        )

    total_violations = sum(result.violation_count for result in rule_results)
    rules_with_violations = sum(
        1 for result in rule_results if result.violation_count > 0
    )

    has_high_severity = any(result.severity == "high" for result in rule_results)

    if total_violations == 0:
        status = "passed"
    elif has_high_severity:
        status = "failed"
    else:
        status = "warning"

    return BusinessRulesCheckResult(
        status=status,
        rules_checked=len(rules),
        rules_with_violations=rules_with_violations,
        total_violations=total_violations,
        results=rule_results,
    )