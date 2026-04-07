import pandas as pd
import numpy as np

TRADING_DAYS = 252

def annualized_return(df: pd.DataFrame) -> float:
    """
    Calculate annualized return from daily returns.
    """
    daily_returns = df["daily_return"].dropna()
    if len(daily_returns) == 0:
        return np.nan

    cumulative_growth = (1 + daily_returns).prod()
    n_days = len(daily_returns)

    return cumulative_growth ** (TRADING_DAYS / n_days) - 1


def annualized_volatility(df: pd.DataFrame) -> float:
    """
    Calculate annualized volatility from daily returns.
    """
    daily_returns = df["daily_return"].dropna()
    if len(daily_returns) == 0:
        return np.nan

    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    """
    ann_ret = annualized_return(df)
    ann_vol = annualized_volatility(df)

    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan

    return (ann_ret - risk_free_rate) / ann_vol