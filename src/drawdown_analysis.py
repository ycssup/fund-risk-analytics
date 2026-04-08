import pandas as pd
import numpy as np


def _ensure_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure drawdown columns exist.
    """
    if {"cum_max", "drawdown"}.issubset(df.columns):
        return df.copy()

    return calculate_drawdown(df)


def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown series based on NAV.

    Returns:
    - DataFrame with 'cum_max' and 'drawdown'
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    # cumulative max NAV
    df["cum_max"] = df["nav"].cummax()

    # drawdown = current NAV / historical max - 1
    df["drawdown"] = df["nav"] / df["cum_max"] - 1

    return df


def max_drawdown(df: pd.DataFrame) -> float:
    """
    Get maximum drawdown value
    """
    df = _ensure_drawdown(df)
    return float(df["drawdown"].min())


def max_drawdown_date(df: pd.DataFrame):
    """
    Get the date when maximum drawdown occurs.
    """
    df = _ensure_drawdown(df)
    if len(df) == 0 or df["drawdown"].dropna().empty:
        return pd.NaT

    trough_idx = df["drawdown"].idxmin()
    if "date" not in df.columns:
        return trough_idx

    return df.loc[trough_idx, "date"]


def max_drawdown_recovery_days(df: pd.DataFrame) -> float:
    """
    Calculate recovery days from maximum drawdown trough to previous NAV high.

    Returns NaN if the maximum drawdown has not recovered by the end of the data.
    """
    df = _ensure_drawdown(df)
    if len(df) == 0 or "date" not in df.columns or df["drawdown"].dropna().empty:
        return np.nan

    trough_idx = df["drawdown"].idxmin()
    peak_nav = df.loc[trough_idx, "cum_max"]
    recovery = df.loc[trough_idx:]
    recovery = recovery[recovery["nav"] >= peak_nav]

    if recovery.empty:
        return np.nan

    recovery_date = recovery.iloc[0]["date"]
    trough_date = df.loc[trough_idx, "date"]

    return float((recovery_date - trough_date).days)


def max_drawdown_details(df: pd.DataFrame) -> dict:
    """
    Summarize maximum drawdown value, occurrence date, and recovery days.
    """
    return {
        "max_drawdown": max_drawdown(df),
        "max_drawdown_date": max_drawdown_date(df),
        "max_drawdown_recovery_days": max_drawdown_recovery_days(df),
    }


def _period_labels(dates: pd.Series, period: str) -> pd.Series:
    if period == "quarterly":
        return dates.dt.to_period("Q").astype(str)

    if period == "semiannual":
        half = np.where(dates.dt.quarter <= 2, "H1", "H2")
        return dates.dt.year.astype(str) + "-" + pd.Series(half, index=dates.index)

    if period == "annual":
        return dates.dt.year.astype(str)

    raise ValueError("period must be one of: quarterly, semiannual, annual")


def drawdown_frequency(df: pd.DataFrame, period: str = "quarterly") -> pd.DataFrame:
    """
    Calculate drawdown frequency by period.

    drawdown_event_count counts new drawdown episodes in the period.
    drawdown_observation_frequency is the share of observations with drawdown < 0.
    """
    df = _ensure_drawdown(df)
    if "date" not in df.columns:
        raise KeyError("drawdown_frequency requires a date column")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "observations",
                "drawdown_observations",
                "drawdown_observation_frequency",
                "drawdown_event_count",
            ]
        )

    data["period"] = _period_labels(data["date"], period)
    data["is_drawdown"] = data["drawdown"] < 0
    data["drawdown_event_start"] = data["is_drawdown"] & ~data["is_drawdown"].shift(fill_value=False)

    frequency = (
        data.groupby("period")
        .agg(
            observations=("is_drawdown", "size"),
            drawdown_observations=("is_drawdown", "sum"),
            drawdown_event_count=("drawdown_event_start", "sum"),
        )
        .reset_index()
    )
    frequency["drawdown_observation_frequency"] = (
        frequency["drawdown_observations"] / frequency["observations"]
    )

    return frequency[
        [
            "period",
            "observations",
            "drawdown_observations",
            "drawdown_observation_frequency",
            "drawdown_event_count",
        ]
    ]


def drawdown_frequency_summary(df: pd.DataFrame) -> dict:
    """
    Calculate quarterly, semiannual, and annual drawdown frequency tables.
    """
    return {
        "quarterly": drawdown_frequency(df, period="quarterly"),
        "semiannual": drawdown_frequency(df, period="semiannual"),
        "annual": drawdown_frequency(df, period="annual"),
    }
