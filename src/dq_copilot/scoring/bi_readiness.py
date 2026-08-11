from dq_copilot.models import (
    BIReadinessScoreComponent,
    BIReadinessScoreResult,
    DataQualityRunResult,
)


def _score_issue_percentage(
    issue_percentage: float,
    max_score: int,
    low_threshold: float,
    medium_threshold: float,
    high_threshold: float,
) -> int:
    """
    Convert an issue percentage into a component score.

    The higher the issue percentage, the lower the score.
    """
    if issue_percentage == 0:
        return max_score

    if issue_percentage <= low_threshold:
        return round(max_score * 0.9)

    if issue_percentage <= medium_threshold:
        return round(max_score * 0.65)

    if issue_percentage <= high_threshold:
        return round(max_score * 0.4)

    return round(max_score * 0.1)


def _component_status(score: int, max_score: int) -> str:
    """
    Classify component health based on its score ratio.
    """
    if max_score == 0:
        return "skipped"

    ratio = score / max_score

    if ratio >= 0.85:
        return "strong"

    if ratio >= 0.6:
        return "acceptable"

    return "weak"


def _rating_from_score(score: int) -> str:
    """
    Convert the overall score into a business-friendly rating.
    """
    if score >= 90:
        return "excellent"

    if score >= 75:
        return "good"

    if score >= 50:
        return "needs_attention"

    return "poor"


def _score_missing_values(result: DataQualityRunResult) -> BIReadinessScoreComponent:
    max_score = 25
    schema = result.schema_profile

    total_cells = schema.row_count * schema.column_count

    if total_cells == 0:
        missing_percentage = 0.0
    else:
        missing_percentage = round(
            (result.missing_values.total_missing_values / total_cells) * 100,
            2,
        )

    score = _score_issue_percentage(
        issue_percentage=missing_percentage,
        max_score=max_score,
        low_threshold=1.0,
        medium_threshold=5.0,
        high_threshold=10.0,
    )

    return BIReadinessScoreComponent(
        component_name="missing_values",
        score=score,
        max_score=max_score,
        status=_component_status(score, max_score),
        explanation=(
            f"Missing values represent {missing_percentage:.2f}% of all dataset cells."
        ),
    )


def _score_duplicates(result: DataQualityRunResult) -> BIReadinessScoreComponent:
    max_score = 25
    duplicate_percentage = result.duplicates.duplicate_percentage

    score = _score_issue_percentage(
        issue_percentage=duplicate_percentage,
        max_score=max_score,
        low_threshold=1.0,
        medium_threshold=5.0,
        high_threshold=10.0,
    )

    return BIReadinessScoreComponent(
        component_name="duplicates",
        score=score,
        max_score=max_score,
        status=_component_status(score, max_score),
        explanation=(
            f"Duplicated rows represent {duplicate_percentage:.2f}% of checked rows."
        ),
    )


def _score_type_validation(result: DataQualityRunResult) -> BIReadinessScoreComponent:
    max_score = 20

    if result.type_validation is None:
        score = 10

        return BIReadinessScoreComponent(
            component_name="type_validation",
            score=score,
            max_score=max_score,
            status="skipped",
            explanation=(
                "Type validation was skipped because no expected schema was provided. "
                "The dataset receives partial credit, but confidence is limited."
            ),
        )

    total_non_null_values = sum(
        column.non_null_count for column in result.type_validation.results
    )

    if total_non_null_values == 0:
        invalid_percentage = 0.0
    else:
        invalid_percentage = round(
            (result.type_validation.total_invalid_values / total_non_null_values)
            * 100,
            2,
        )

    score = _score_issue_percentage(
        issue_percentage=invalid_percentage,
        max_score=max_score,
        low_threshold=1.0,
        medium_threshold=5.0,
        high_threshold=10.0,
    )

    return BIReadinessScoreComponent(
        component_name="type_validation",
        score=score,
        max_score=max_score,
        status=_component_status(score, max_score),
        explanation=(
            f"Invalid type values represent {invalid_percentage:.2f}% "
            "of checked non-null values."
        ),
    )


def _score_business_rules(result: DataQualityRunResult) -> BIReadinessScoreComponent:
    max_score = 20

    if result.business_rules is None:
        score = 10

        return BIReadinessScoreComponent(
            component_name="business_rules",
            score=score,
            max_score=max_score,
            status="skipped",
            explanation=(
                "Business-rule checks were skipped because no rules were configured. "
                "The dataset receives partial credit, but business validity is not fully proven."
            ),
        )

    denominator = (
        result.schema_profile.row_count * result.business_rules.rules_checked
    )

    if denominator == 0:
        violation_percentage = 0.0
    else:
        violation_percentage = round(
            (result.business_rules.total_violations / denominator) * 100,
            2,
        )

    score = _score_issue_percentage(
        issue_percentage=violation_percentage,
        max_score=max_score,
        low_threshold=1.0,
        medium_threshold=5.0,
        high_threshold=10.0,
    )

    return BIReadinessScoreComponent(
        component_name="business_rules",
        score=score,
        max_score=max_score,
        status=_component_status(score, max_score),
        explanation=(
            f"Business-rule violations represent {violation_percentage:.2f}% "
            "of all evaluated rule-row combinations."
        ),
    )


def _score_check_coverage(result: DataQualityRunResult) -> BIReadinessScoreComponent:
    max_score = 10

    has_type_validation = result.type_validation is not None
    has_business_rules = result.business_rules is not None

    if has_type_validation and has_business_rules:
        score = 10
        explanation = (
            "Core checks, type validation, and business rules were all executed."
        )
    elif has_type_validation or has_business_rules:
        score = 7
        explanation = (
            "Core checks were executed, but only one advanced validation layer was configured."
        )
    else:
        score = 5
        explanation = (
            "Only core checks were executed. Add expected schema and business rules "
            "to improve BI-readiness confidence."
        )

    return BIReadinessScoreComponent(
        component_name="check_coverage",
        score=score,
        max_score=max_score,
        status=_component_status(score, max_score),
        explanation=explanation,
    )


def _build_recommendations(
    result: DataQualityRunResult,
    breakdown: list[BIReadinessScoreComponent],
) -> list[str]:
    recommendations: list[str] = []

    weak_components = [
        component.component_name
        for component in breakdown
        if component.status == "weak"
    ]

    if "missing_values" in weak_components:
        recommendations.append(
            "Investigate columns with missing values and decide whether to fix them at the source, impute them, or exclude them from BI metrics."
        )

    if "duplicates" in weak_components:
        recommendations.append(
            "Review duplicate rows or duplicate keys because they can inflate KPIs and break joins."
        )

    if result.type_validation is None:
        recommendations.append(
            "Define an expected schema to validate important columns such as IDs, dates, numeric measures, and flags."
        )
    elif "type_validation" in weak_components:
        recommendations.append(
            "Correct invalid type values before loading the dataset into analytical models or dashboards."
        )

    if result.business_rules is None:
        recommendations.append(
            "Add business rules for critical fields, such as required IDs, valid date ranges, positive amounts, or allowed categories."
        )
    elif "business_rules" in weak_components:
        recommendations.append(
            "Review business-rule violations with business stakeholders and update the data contract if needed."
        )

    if not recommendations:
        recommendations.append(
            "The dataset looks reasonably ready for BI based on the configured checks."
        )

    return recommendations


def compute_bi_readiness_score(
    result: DataQualityRunResult,
) -> BIReadinessScoreResult:
    """
    Compute a deterministic BI-readiness score from a data quality run result.

    The score is explainable and based only on deterministic check outputs.
    """
    breakdown = [
        _score_missing_values(result),
        _score_duplicates(result),
        _score_type_validation(result),
        _score_business_rules(result),
        _score_check_coverage(result),
    ]

    overall_score = sum(component.score for component in breakdown)
    rating = _rating_from_score(overall_score)

    summary = (
        f"The dataset received a BI-readiness score of {overall_score}/100 "
        f"with rating '{rating}'."
    )

    recommendations = _build_recommendations(
        result=result,
        breakdown=breakdown,
    )

    return BIReadinessScoreResult(
        overall_score=overall_score,
        rating=rating,
        summary=summary,
        breakdown=breakdown,
        recommendations=recommendations,
    )