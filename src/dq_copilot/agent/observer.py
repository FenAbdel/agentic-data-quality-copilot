from dq_copilot.agent.planning_models import CopilotObservation
from dq_copilot.models import DataQualityRunResult, VerificationRunResult


def build_copilot_observations(
    result: DataQualityRunResult,
    verification_result: VerificationRunResult | None = None,
) -> list[CopilotObservation]:
    """
    Build high-level observations from deterministic check results.

    These observations are not LLM-generated.
    They are derived from verified tool outputs.
    """
    observations: list[CopilotObservation] = []

    if result.missing_values.total_missing_values > 0:
        observations.append(
            CopilotObservation(
                source="missing_values",
                severity="warning",
                message=(
                    f"Found {result.missing_values.total_missing_values} missing values "
                    f"across {result.missing_values.columns_with_missing_values} columns."
                ),
            )
        )
    else:
        observations.append(
            CopilotObservation(
                source="missing_values",
                severity="info",
                message="No missing values were detected.",
            )
        )

    if result.duplicates.duplicate_row_count > 0:
        severity = "critical" if result.duplicates.status == "failed" else "warning"
        observations.append(
            CopilotObservation(
                source="duplicates",
                severity=severity,
                message=(
                    f"Detected {result.duplicates.duplicate_row_count} duplicated rows "
                    f"using scope `{result.duplicates.scope}`."
                ),
            )
        )
    else:
        observations.append(
            CopilotObservation(
                source="duplicates",
                severity="info",
                message="No duplicated rows were detected for the configured duplicate scope.",
            )
        )

    if result.type_validation is None:
        observations.append(
            CopilotObservation(
                source="type_validation",
                severity="warning",
                message="Type validation was skipped because no expected schema was configured.",
            )
        )
    elif result.type_validation.total_invalid_values > 0:
        observations.append(
            CopilotObservation(
                source="type_validation",
                severity="warning",
                message=(
                    f"Found {result.type_validation.total_invalid_values} invalid type values "
                    f"across {result.type_validation.columns_with_invalid_values} columns."
                ),
            )
        )
    else:
        observations.append(
            CopilotObservation(
                source="type_validation",
                severity="info",
                message="No invalid type values were detected for the configured schema.",
            )
        )

    if result.business_rules is None:
        observations.append(
            CopilotObservation(
                source="business_rules",
                severity="warning",
                message="Business-rule checks were skipped because no rules were configured.",
            )
        )
    elif result.business_rules.total_violations > 0:
        observations.append(
            CopilotObservation(
                source="business_rules",
                severity="warning",
                message=(
                    f"Found {result.business_rules.total_violations} business-rule violations "
                    f"across {result.business_rules.rules_with_violations} rules."
                ),
            )
        )
    else:
        observations.append(
            CopilotObservation(
                source="business_rules",
                severity="info",
                message="No business-rule violations were detected for the configured rules.",
            )
        )

    if result.sql_analysis and result.sql_analysis.queries_failed > 0:
        observations.append(
            CopilotObservation(
                source="sql_analysis",
                severity="warning",
                message=(
                    f"{result.sql_analysis.queries_failed} DuckDB SQL query/queries failed. "
                    "Review SQL analysis limitations in the report."
                ),
            )
        )

    if result.bi_readiness_score is not None:
        score = result.bi_readiness_score

        if score.rating in {"poor", "needs_attention"}:
            severity = "critical" if score.rating == "poor" else "warning"
        else:
            severity = "info"

        observations.append(
            CopilotObservation(
                source="bi_readiness_score",
                severity=severity,
                message=(
                    f"BI-readiness score is {score.overall_score}/{score.max_score} "
                    f"with rating `{score.rating}`."
                ),
            )
        )

    if verification_result is not None:
        severity = "info"

        if verification_result.status == "warning":
            severity = "warning"
        elif verification_result.status == "failed":
            severity = "critical"

        observations.append(
            CopilotObservation(
                source="result_verification",
                severity=severity,
                message=(
                    f"Result verification finished with status `{verification_result.status}` "
                    f"({verification_result.checks_failed} failed, "
                    f"{verification_result.checks_warning} warning)."
                ),
            )
        )

    return observations