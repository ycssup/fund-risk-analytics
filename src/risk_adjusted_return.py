import pandas as pd
import numpy as np
from typing import Optional

from src.drawdown_analysis import max_drawdown
from src.frequency import TRADING_DAYS, infer_periods_per_year
from src.risk_metrics import annualized_return, annualized_volatility


def _get_return_series(df: pd.DataFrame, return_col: str = "daily_return") -> pd.Series:
    if return_col not in df.columns:
        raise KeyError(f"Missing return column: {return_col}")

    return pd.to_numeric(df[return_col], errors="coerce").dropna()


def _periodic_rate(annual_rate: float, periods_per_year: float) -> float:
    if periods_per_year <= 0:
        return np.nan

    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def _aligned_fund_benchmark_returns(
    df: pd.DataFrame,
    fund_return_col: str = "daily_return",
    benchmark_return_col: str = "benchmark_return",
) -> pd.DataFrame:
    if fund_return_col not in df.columns or benchmark_return_col not in df.columns:
        return pd.DataFrame(columns=["fund_return", "benchmark_return"])

    aligned = pd.DataFrame(
        {
            "fund_return": pd.to_numeric(df[fund_return_col], errors="coerce"),
            "benchmark_return": pd.to_numeric(df[benchmark_return_col], errors="coerce"),
        }
    ).dropna()

    return aligned


def _beta_from_returns(aligned_returns: pd.DataFrame) -> float:
    if len(aligned_returns) < 2:
        return np.nan

    benchmark_variance = aligned_returns["benchmark_return"].var()
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan

    return float(
        aligned_returns["fund_return"].cov(aligned_returns["benchmark_return"])
        / benchmark_variance
    )


def sharpe_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    risk_free_rate is annualized. The default is 2%.
    """
    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(df, periods_per_year=periods_per_year)

    if ann_vol == 0 or np.isnan(ann_vol) or np.isnan(ann_ret):
        return np.nan

    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    df: pd.DataFrame,
    target_return: float = 0.02,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate annualized Sortino ratio.

    target_return is annualized. The default is 2%.
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(df)

    returns = _get_return_series(df, return_col=return_col)
    if len(returns) == 0:
        return np.nan

    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    target_period_return = _periodic_rate(target_return, periods_per_year)
    downside_returns = np.minimum(returns - target_period_return, 0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year)

    if downside_deviation == 0 or np.isnan(downside_deviation) or np.isnan(ann_ret):
        return np.nan

    return float((ann_ret - target_return) / downside_deviation)


def calmar_ratio(df: pd.DataFrame, periods_per_year: Optional[float] = None) -> float:
    """
    Calculate Calmar ratio as annualized return / absolute maximum drawdown.
    """
    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    mdd = max_drawdown(df)

    if mdd == 0 or np.isnan(mdd) or np.isnan(ann_ret):
        return np.nan

    return float(ann_ret / abs(mdd))


def treynor_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    beta_value: Optional[float] = None,
    benchmark_return_col: str = "benchmark_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate Treynor ratio as annualized excess return / beta.

    If beta_value is not provided, df must contain benchmark_return_col.
    """
    ann_ret = annualized_return(df, periods_per_year=periods_per_year)

    if beta_value is None:
        aligned_returns = _aligned_fund_benchmark_returns(
            df,
            benchmark_return_col=benchmark_return_col,
        )
        beta_value = _beta_from_returns(aligned_returns)

    if beta_value == 0 or np.isnan(beta_value) or np.isnan(ann_ret):
        return np.nan

    return float((ann_ret - risk_free_rate) / beta_value)


def treynor_black_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    fund_return_col: str = "daily_return",
    benchmark_return_col: str = "benchmark_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate Treynor-Black appraisal ratio as annualized alpha / residual volatility.

    df must contain both fund_return_col and benchmark_return_col.
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(df)

    aligned_returns = _aligned_fund_benchmark_returns(
        df,
        fund_return_col=fund_return_col,
        benchmark_return_col=benchmark_return_col,
    )
    if len(aligned_returns) < 3:
        return np.nan

    risk_free_period = _periodic_rate(risk_free_rate, periods_per_year)
    fund_excess = aligned_returns["fund_return"] - risk_free_period
    benchmark_excess = aligned_returns["benchmark_return"] - risk_free_period

    benchmark_variance = benchmark_excess.var()
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan

    beta_value = fund_excess.cov(benchmark_excess) / benchmark_variance
    alpha_period = fund_excess.mean() - beta_value * benchmark_excess.mean()
    residuals = fund_excess - (alpha_period + beta_value * benchmark_excess)
    residual_volatility = residuals.std() * np.sqrt(periods_per_year)

    if residual_volatility == 0 or np.isnan(residual_volatility):
        return np.nan

    annualized_alpha = alpha_period * periods_per_year

    return float(annualized_alpha / residual_volatility)


def risk_adjusted_return_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    benchmark_return_col: str = "benchmark_return",
    periods_per_year: Optional[float] = None,
) -> dict:
    """
    Calculate risk-adjusted return metrics.
    """
    return {
        "sharpe_ratio": sharpe_ratio(
            df,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "sortino_ratio": sortino_ratio(
            df,
            target_return=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "calmar_ratio": calmar_ratio(df, periods_per_year=periods_per_year),
        "treynor_ratio": treynor_ratio(
            df,
            risk_free_rate=risk_free_rate,
            benchmark_return_col=benchmark_return_col,
            periods_per_year=periods_per_year,
        ),
        "treynor_black_ratio": treynor_black_ratio(
            df,
            risk_free_rate=risk_free_rate,
            benchmark_return_col=benchmark_return_col,
            periods_per_year=periods_per_year,
        ),
    }
