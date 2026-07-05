import pandas as pd

from dq_copilot.models import ColumnSchemaProfile, DatasetSchemaProfile


def _get_sample_values(series: pd.Series, max_values: int = 5) -> list[str]:
    """
    Return a small list of non-null sample values as strings.

    We convert values to strings because reports and UI displays
    should not depend on NumPy/Pandas internal types.
    """
    values = series.dropna().unique().tolist()
    return [str(value) for value in values[:max_values]]


def profile_schema(
    dataframe: pd.DataFrame,
    dataset_name: str = "dataset",
) -> DatasetSchemaProfile:
    """
    Build a basic schema profile for a Pandas DataFrame.

    Parameters
    ----------
    dataframe:
        Dataset to profile.

    dataset_name:
        Human-readable dataset name.

    Returns
    -------
    DatasetSchemaProfile
        Structured schema profile.
    """
    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    columns: list[ColumnSchemaProfile] = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]

        null_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())

        if row_count == 0:
            null_percentage = 0.0
        else:
            null_percentage = round((null_count / row_count) * 100, 2)

        columns.append(
            ColumnSchemaProfile(
                name=str(column_name),
                pandas_dtype=str(series.dtype),
                non_null_count=non_null_count,
                null_count=null_count,
                null_percentage=null_percentage,
                sample_values=_get_sample_values(series),
            )
        )

    return DatasetSchemaProfile(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=column_count,
        columns=columns,
    )