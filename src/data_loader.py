import pandas as pd


def load_nav_data(file_path: str) -> pd.DataFrame:
    """
    Load NAV data from CSV or Excel.

    Assumptions:
    - First column = date
    - Second column = nav
    """

    # Read file based on extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    # Keep only the first two columns
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "nav"]

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["date", "nav"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df