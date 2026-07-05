from pathlib import Path

from dq_copilot.loaders.csv_loader import load_csv
from dq_copilot.profiling.schema_profiler import profile_schema


SAMPLE_CSV_PATH = Path("data/samples/customers.csv")


def test_load_csv_returns_dataframe():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    assert dataframe is not None
    assert len(dataframe) == 5
    assert len(dataframe.columns) == 6


def test_profile_schema_returns_dataset_summary():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    profile = profile_schema(
        dataframe=dataframe,
        dataset_name=SAMPLE_CSV_PATH.name,
    )

    assert profile.dataset_name == "customers.csv"
    assert profile.row_count == 5
    assert profile.column_count == 6
    assert len(profile.columns) == 6


def test_profile_schema_detects_missing_values():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    profile = profile_schema(
        dataframe=dataframe,
        dataset_name=SAMPLE_CSV_PATH.name,
    )

    columns_by_name = {column.name: column for column in profile.columns}

    assert columns_by_name["email"].null_count == 1
    assert columns_by_name["email"].null_percentage == 20.0

    assert columns_by_name["age"].null_count == 1
    assert columns_by_name["age"].null_percentage == 20.0


def test_profile_schema_detects_pandas_dtypes():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    profile = profile_schema(
        dataframe=dataframe,
        dataset_name=SAMPLE_CSV_PATH.name,
    )

    columns_by_name = {column.name: column for column in profile.columns}

    assert columns_by_name["customer_id"].pandas_dtype == "int64"
    assert columns_by_name["customer_name"].pandas_dtype == "object"
    assert columns_by_name["email"].pandas_dtype == "object"