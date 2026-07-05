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