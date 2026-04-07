import pandas as pd

def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns based on NAV.

    Parameters:
    - df: DataFrame with 'nav' column

    Returns:
    - DataFrame with an added 'daily_return' column
    """
    df = df.copy()
    df["daily_return"] = df["nav"].pct_change()
    return df


def calculate_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns based on daily returns.

    Parameters:
    - df: DataFrame with 'daily_return' column

    Returns:
    - DataFrame with an added 'cumulative_return' column
    """
    df = df.copy()
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df