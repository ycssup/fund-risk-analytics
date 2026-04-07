import pandas as pd
import numpy as np

TRADING_DAYS = 252

def rolling_volatility(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    Rolling volatility
    """
    return df["daily_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    Rolling Sharpe ratio
    """
    rolling_mean = df["daily_return"].rolling(window).mean() * TRADING_DAYS
    rolling_std = df["daily_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)

    return rolling_mean / rolling_std