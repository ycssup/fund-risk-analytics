import pandas as pd


def load_nav_data(file_path: str) -> pd.DataFrame:
    """
    Load NAV data from CSV or Excel.

    Assumptions:
    - First column = date
    - Second column = nav
    """

    file_path = str(file_path)
    file_extension = file_path.lower().rsplit(".", 1)[-1]

    # Read file based on extension
    if file_extension == "csv":
        df = pd.read_csv(file_path)
    elif file_extension in {"xlsx", "xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV, XLSX, or XLS.")

    if df.shape[1] < 2:
        raise ValueError("NAV data must contain at least two columns: date and nav.")

    # First column = date, second column = nav. Preserve extra columns such as benchmark_return.
    df = df.copy()
    columns = list(df.columns)
    columns[0] = "date"
    columns[1] = "nav"
    df.columns = columns

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["date", "nav"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df
