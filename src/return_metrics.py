import pandas as pd
import numpy as np
from typing import Dict, Tuple

CALENDAR_DAYS = 365.25


def _prepare_nav_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare NAV data sorted by date.
    """
    required_columns = {"date", "nav"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    nav_data = df.copy()
    nav_data["date"] = pd.to_datetime(nav_data["date"], errors="coerce")
    nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
    nav_data = nav_data.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)

    return nav_data


def _total_return(start_nav: float, end_nav: float) -> float:
    if start_nav <= 0 or end_nav <= 0:
        return np.nan

    return float(end_nav / start_nav - 1)


def calculate_periodic_returns(
    df: pd.DataFrame,
    return_col: str = "daily_return",
) -> pd.DataFrame:
    """
    Calculate periodic returns based on NAV.

    The default column name remains daily_return for compatibility with existing
    risk and rolling metric functions.
    """
    df = _prepare_nav_data(df)
    df[return_col] = df["nav"].pct_change()

    return df


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias for calculate_periodic_returns.
    """
    return calculate_periodic_returns(df, return_col="daily_return")


def calculate_cumulative_returns(
    df: pd.DataFrame,
    return_col: str = "daily_return",
    cumulative_return_col: str = "cumulative_return",
) -> pd.DataFrame:
    """
    Calculate cumulative returns based on periodic returns.
    """
    df = df.copy()
    if return_col not in df.columns:
        df = calculate_periodic_returns(df, return_col=return_col)

    df[cumulative_return_col] = (1 + df[return_col]).cumprod() - 1

    return df


def inception_return(df: pd.DataFrame) -> float:
    """
    Calculate total return since inception.
    """
    nav_data = _prepare_nav_data(df)
    if len(nav_data) < 2:
        return np.nan

    return _total_return(nav_data["nav"].iloc[0], nav_data["nav"].iloc[-1])


def inception_annualized_return(df: pd.DataFrame) -> float:
    """
    Calculate annualized return since inception using CAGR.
    """
    nav_data = _prepare_nav_data(df)
    if len(nav_data) < 2:
        return np.nan

    start_date = nav_data["date"].iloc[0]
    end_date = nav_data["date"].iloc[-1]
    elapsed_days = (end_date - start_date).days
    if elapsed_days <= 0:
        return np.nan

    total_return = inception_return(nav_data)
    if np.isnan(total_return) or total_return <= -1:
        return np.nan

    return float((1 + total_return) ** (CALENDAR_DAYS / elapsed_days) - 1)


def year_to_date_return(df: pd.DataFrame) -> float:
    """
    Calculate year-to-date return.

    Uses the previous year-end NAV if available. If the series starts in the
    current year, uses the first NAV in the current year as the starting NAV.
    """
    nav_data = _prepare_nav_data(df)
    if len(nav_data) < 2:
        return np.nan

    latest_date = nav_data["date"].iloc[-1]
    latest_nav = nav_data["nav"].iloc[-1]
    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)

    previous_year_data = nav_data[nav_data["date"] < year_start]
    if not previous_year_data.empty:
        start_nav = previous_year_data["nav"].iloc[-1]
    else:
        current_year_data = nav_data[nav_data["date"].dt.year == latest_date.year]
        if current_year_data.empty:
            return np.nan
        start_nav = current_year_data["nav"].iloc[0]

    return _total_return(start_nav, latest_nav)


def one_year_return(df: pd.DataFrame) -> float:
    """
    Calculate trailing one-year return.
    """
    nav_data = _prepare_nav_data(df)
    if len(nav_data) < 2:
        return np.nan

    latest_date = nav_data["date"].iloc[-1]
    latest_nav = nav_data["nav"].iloc[-1]
    lookback_date = latest_date - pd.DateOffset(years=1)
    lookback_data = nav_data[nav_data["date"] <= lookback_date]

    if lookback_data.empty:
        return np.nan

    start_nav = lookback_data["nav"].iloc[-1]

    return _total_return(start_nav, latest_nav)


def win_rate(df: pd.DataFrame, return_col: str = "daily_return") -> float:
    """
    Calculate the share of positive return observations.
    """
    data = df.copy()
    if return_col not in data.columns:
        data = calculate_periodic_returns(data, return_col=return_col)

    returns = pd.to_numeric(data[return_col], errors="coerce").dropna()
    if len(returns) == 0:
        return np.nan

    return float((returns > 0).sum() / len(returns))


def monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate calendar monthly returns.

    The first month uses first observed NAV in that month as the starting NAV.
    Subsequent months use previous month-end NAV.
    """
    nav_data = _prepare_nav_data(df)
    if nav_data.empty:
        return pd.DataFrame(columns=["month", "monthly_return"])

    nav_data["month"] = nav_data["date"].dt.to_period("M")
    month_end_nav = nav_data.groupby("month")["nav"].last()
    month_start_nav = nav_data.groupby("month")["nav"].first()
    returns = month_end_nav.pct_change()

    if len(returns) > 0:
        first_month = returns.index[0]
        returns.iloc[0] = _total_return(month_start_nav.loc[first_month], month_end_nav.loc[first_month])

    return returns.rename("monthly_return").reset_index().assign(month=lambda x: x["month"].astype(str))


def annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate calendar annual returns.

    For years after inception, annual return is year-end NAV / previous year-end
    NAV - 1. For the inception year, it is inception-year-end NAV / inception
    NAV - 1.
    """
    nav_data = _prepare_nav_data(df)
    if nav_data.empty:
        return pd.DataFrame(columns=["year", "annual_return"])

    nav_data["year"] = nav_data["date"].dt.year
    year_end_nav = nav_data.groupby("year")["nav"].last()
    first_year_nav = nav_data.groupby("year")["nav"].first()
    returns = year_end_nav.pct_change()

    if len(returns) > 0:
        first_year = returns.index[0]
        returns.iloc[0] = _total_return(first_year_nav.loc[first_year], year_end_nav.loc[first_year])

    return returns.rename("annual_return").reset_index()


def return_skewness(df: pd.DataFrame, return_col: str = "daily_return") -> float:
    """
    Calculate skewness of return distribution.
    """
    data = df.copy()
    if return_col not in data.columns:
        data = calculate_periodic_returns(data, return_col=return_col)

    returns = pd.to_numeric(data[return_col], errors="coerce").dropna()
    if len(returns) < 3:
        return np.nan

    return float(returns.skew())


def consecutive_month_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate current and maximum consecutive up/down month counts.
    """
    monthly = monthly_returns(df)
    if monthly.empty:
        return {
            "current_consecutive_up_months": 0,
            "current_consecutive_down_months": 0,
            "max_consecutive_up_months": 0,
            "max_consecutive_down_months": 0,
        }

    returns = pd.to_numeric(monthly["monthly_return"], errors="coerce").dropna()

    current_up = 0
    current_down = 0
    max_up = 0
    max_down = 0
    running_up = 0
    running_down = 0

    for value in returns:
        if value > 0:
            running_up += 1
            running_down = 0
        elif value < 0:
            running_down += 1
            running_up = 0
        else:
            running_up = 0
            running_down = 0

        max_up = max(max_up, running_up)
        max_down = max(max_down, running_down)

    for value in reversed(returns.tolist()):
        if value > 0 and current_down == 0:
            current_up += 1
        elif value < 0 and current_up == 0:
            current_down += 1
        else:
            break

    return {
        "current_consecutive_up_months": current_up,
        "current_consecutive_down_months": current_down,
        "max_consecutive_up_months": max_up,
        "max_consecutive_down_months": max_down,
    }


def return_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate headline return metrics.
    """
    metrics = {
        "inception_annualized_return": inception_annualized_return(df),
        "inception_return": inception_return(df),
        "year_to_date_return": year_to_date_return(df),
        "one_year_return": one_year_return(df),
        "win_rate": win_rate(df),
        "return_skewness": return_skewness(df),
    }
    metrics.update(consecutive_month_counts(df))

    return metrics


def return_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return monthly and annual return tables.
    """
    return monthly_returns(df), annual_returns(df)
