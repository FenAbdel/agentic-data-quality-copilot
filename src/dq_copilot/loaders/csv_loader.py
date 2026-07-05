from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the file is not a CSV file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")

    dataframe = pd.read_csv(path)

    # Pandas 3 can infer text columns as the dedicated "str" dtype. Keep CSV
    # ingestion on classic pandas dtypes so profiling output is version-stable.
    for column_name in dataframe.columns:
        if str(dataframe[column_name].dtype) == "str":
            dataframe[column_name] = dataframe[column_name].astype("object")

    return dataframe
