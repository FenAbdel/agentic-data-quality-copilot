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