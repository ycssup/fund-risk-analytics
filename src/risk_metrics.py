import pandas as pd
import numpy as np
from typing import Dict, Optional, Sequence
from src.frequency import CALENDAR_DAYS, TRADING_DAYS, infer_periods_per_year

DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)
DEFAULT_HOLDING_PERIOD_DAYS = (1, 10)


def _infer_periods_per_year(df: pd.DataFrame) -> float:
    """
    Infer the number of return observations per year from the date column.
    """
    return infer_periods_per_year(df)


def _get_return_series(df: pd.DataFrame, return_col: str = "daily_return") -> pd.Series:
    """
    Return a clean numeric return series.
    """
    if return_col not in df.columns:
        raise KeyError(f"Missing return column: {return_col}")

    return pd.to_numeric(df[return_col], errors="coerce").dropna()


def _horizon_scale(
    holding_period_days: int = 1,
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Convert the return observation frequency to the requested trading-day horizon.
    """
    if periods_per_year is None:
        periods_per_year = TRADING_DAYS

    if holding_period_days <= 0 or periods_per_year <= 0:
        return np.nan

    horizon_periods = holding_period_days * periods_per_year / TRADING_DAYS
    return np.sqrt(horizon_periods)


def _format_confidence_level(confidence_level: float) -> str:
    return f"{int(round(confidence_level * 100))}"


def annualized_return(df: pd.DataFrame, periods_per_year: Optional[float] = None) -> float:
    """
    Calculate annualized return using CAGR from NAV and actual date range.
    """
    if {"date", "nav"}.issubset(df.columns):
        nav_data = df.dropna(subset=["date", "nav"]).sort_values("date")
        nav_data = nav_data.copy()
        nav_data["date"] = pd.to_datetime(nav_data["date"], errors="coerce")
        nav_data = nav_data.dropna(subset=["date", "nav"]).sort_values("date")
        if len(nav_data) < 2:
            return np.nan

        start_nav = nav_data["nav"].iloc[0]
        end_nav = nav_data["nav"].iloc[-1]
        elapsed_days = (nav_data["date"].iloc[-1] - nav_data["date"].iloc[0]).days

        if start_nav <= 0 or end_nav <= 0 or elapsed_days <= 0:
            return np.nan

        return (end_nav / start_nav) ** (CALENDAR_DAYS / elapsed_days) - 1

    returns = _get_return_series(df)
    if len(returns) == 0:
        return np.nan

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df)

    cumulative_growth = (1 + returns).prod()
    if cumulative_growth <= 0 or periods_per_year <= 0:
        return np.nan

    return cumulative_growth ** (periods_per_year / len(returns)) - 1


def annualized_volatility(df: pd.DataFrame, periods_per_year: Optional[float] = None) -> float:
    """
    Calculate annualized volatility from periodic returns.
    """
    returns = _get_return_series(df)
    if len(returns) < 2:
        return np.nan

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df)

    if periods_per_year <= 0:
        return np.nan

    return returns.std() * np.sqrt(periods_per_year)


def value_at_risk(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate historical VaR as a positive loss number.
    """
    returns = _get_return_series(df, return_col=return_col)
    if len(returns) == 0 or not 0 < confidence_level < 1:
        return np.nan

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df)

    quantile = returns.quantile(1 - confidence_level)
    scale = _horizon_scale(
        holding_period_days=holding_period_days,
        periods_per_year=periods_per_year,
    )

    if np.isnan(scale):
        return np.nan

    return float(max(-quantile, 0) * scale)


def conditional_value_at_risk(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate historical CVaR / Expected Shortfall as a positive loss number.
    """
    returns = _get_return_series(df, return_col=return_col)
    if len(returns) == 0 or not 0 < confidence_level < 1:
        return np.nan

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df)

    quantile = returns.quantile(1 - confidence_level)
    tail_returns = returns[returns <= quantile]
    if len(tail_returns) == 0:
        return np.nan

    scale = _horizon_scale(
        holding_period_days=holding_period_days,
        periods_per_year=periods_per_year,
    )

    if np.isnan(scale):
        return np.nan

    return float(max(-tail_returns.mean(), 0) * scale)


def expected_shortfall(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Alias for conditional_value_at_risk.
    """
    return conditional_value_at_risk(
        df=df,
        confidence_level=confidence_level,
        holding_period_days=holding_period_days,
        return_col=return_col,
        periods_per_year=periods_per_year,
    )


def var_metrics(
    df: pd.DataFrame,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    holding_period_days: Sequence[int] = DEFAULT_HOLDING_PERIOD_DAYS,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate VaR metrics for multiple confidence levels and horizons.
    """
    return {
        f"var_{_format_confidence_level(confidence_level)}_{horizon}d": value_at_risk(
            df=df,
            confidence_level=confidence_level,
            holding_period_days=horizon,
            return_col=return_col,
            periods_per_year=periods_per_year,
        )
        for confidence_level in confidence_levels
        for horizon in holding_period_days
    }


def cvar_es_metrics(
    df: pd.DataFrame,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    holding_period_days: Sequence[int] = DEFAULT_HOLDING_PERIOD_DAYS,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate CVaR / Expected Shortfall metrics for multiple confidence levels and horizons.
    """
    return {
        f"cvar_es_{_format_confidence_level(confidence_level)}_{horizon}d": (
            conditional_value_at_risk(
                df=df,
                confidence_level=confidence_level,
                holding_period_days=horizon,
                return_col=return_col,
                periods_per_year=periods_per_year,
            )
        )
        for confidence_level in confidence_levels
        for horizon in holding_period_days
    }


def tail_risk_metrics(
    df: pd.DataFrame,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    holding_period_days: Sequence[int] = DEFAULT_HOLDING_PERIOD_DAYS,
    return_col: str = "daily_return",
    periods_per_year: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate VaR and CVaR / Expected Shortfall metrics.
    """
    metrics = var_metrics(
        df=df,
        confidence_levels=confidence_levels,
        holding_period_days=holding_period_days,
        return_col=return_col,
        periods_per_year=periods_per_year,
    )
    metrics.update(
        cvar_es_metrics(
            df=df,
            confidence_levels=confidence_levels,
            holding_period_days=holding_period_days,
            return_col=return_col,
            periods_per_year=periods_per_year,
        )
    )
    return metrics
