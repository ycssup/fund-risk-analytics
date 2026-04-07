import pandas as pd

def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown series based on NAV.

    Returns:
    - DataFrame with 'cum_max' and 'drawdown'
    """
    df = df.copy()

    # cumulative max NAV
    df["cum_max"] = df["nav"].cummax()

    # drawdown = current NAV / historical max - 1
    df["drawdown"] = df["nav"] / df["cum_max"] - 1

    return df


def max_drawdown(df: pd.DataFrame) -> float:
    """
    Get maximum drawdown value
    """
    return df["drawdown"].min()