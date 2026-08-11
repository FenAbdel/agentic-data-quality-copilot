from typing import Literal

from pydantic import BaseModel, Field


class ColumnSchemaProfile(BaseModel):
    name: str
    pandas_dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    sample_values: list[str] = Field(default_factory=list)


class DatasetSchemaProfile(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    columns: list[ColumnSchemaProfile]


class ColumnMissingValueResult(BaseModel):
    column_name: str
    null_count: int
    null_percentage: float
    severity: Literal["ok", "low", "medium", "high"]


class MissingValuesCheckResult(BaseModel):
    check_name: str = "missing_values"
    status: Literal["passed", "warning", "failed"]
    total_missing_values: int
    columns_checked: int
    columns_with_missing_values: int
    results: list[ColumnMissingValueResult]


class DuplicateGroupResult(BaseModel):
    duplicate_key: dict[str, str]
    duplicate_count: int


class DuplicateCheckResult(BaseModel):
    check_name: str = "duplicate_detection"
    status: Literal["passed", "warning", "failed"]
    scope: Literal["full_row", "key_columns"]
    key_columns: list[str]
    row_count: int
    duplicate_row_count: int
    duplicate_percentage: float
    duplicate_group_count: int
    severity: Literal["ok", "low", "medium", "high"]
    duplicate_groups: list[DuplicateGroupResult] = Field(default_factory=list)


ExpectedDataType = Literal[
    "integer",
    "float",
    "numeric",
    "date",
    "string",
    "boolean",
]


class InvalidTypeValueExample(BaseModel):
    row_index: str
    value: str
    expected_type: ExpectedDataType


class ColumnTypeValidationResult(BaseModel):
    column_name: str
    expected_type: ExpectedDataType
    pandas_dtype: str
    non_null_count: int
    invalid_count: int
    invalid_percentage: float
    severity: Literal["ok", "low", "medium", "high"]
    invalid_examples: list[InvalidTypeValueExample] = Field(default_factory=list)


class TypeValidationCheckResult(BaseModel):
    check_name: str = "type_validation"
    status: Literal["passed", "warning", "failed"]
    columns_checked: int
    columns_with_invalid_values: int
    total_invalid_values: int
    results: list[ColumnTypeValidationResult]


BusinessRuleType = Literal[
    "not_null",
    "range",
    "positive",
    "non_negative",
    "allowed_values",
    "not_future_date",
]


class BusinessRuleConfig(BaseModel):
    rule_name: str
    column_name: str
    rule_type: BusinessRuleType
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    allow_null: bool = True
    description: str | None = None


class BusinessRuleViolationExample(BaseModel):
    row_index: str
    value: str
    message: str


class BusinessRuleResult(BaseModel):
    rule_name: str
    column_name: str
    rule_type: BusinessRuleType
    status: Literal["passed", "failed"]
    violation_count: int
    violation_percentage: float
    severity: Literal["ok", "low", "medium", "high"]
    violation_examples: list[BusinessRuleViolationExample] = Field(default_factory=list)


class BusinessRulesCheckResult(BaseModel):
    check_name: str = "business_rules"
    status: Literal["passed", "warning", "failed"]
    rules_checked: int
    rules_with_violations: int
    total_violations: int
    results: list[BusinessRuleResult]

class CheckExecutionLog(BaseModel):
    step_name: str
    status: Literal["completed", "skipped"]
    message: str


class DataQualityRunConfig(BaseModel):
    dataset_name: str = "dataset"
    duplicate_key_columns: list[str] | None = None
    expected_schema: dict[str, ExpectedDataType] | None = None
    business_rules: list[BusinessRuleConfig] = Field(default_factory=list)


class DataQualityRunResult(BaseModel):
    dataset_name: str
    schema_profile: DatasetSchemaProfile
    missing_values: MissingValuesCheckResult
    duplicates: DuplicateCheckResult
    type_validation: TypeValidationCheckResult | None = None
    business_rules: BusinessRulesCheckResult | None = None
    action_log: list[CheckExecutionLog] = Field(default_factory=list)