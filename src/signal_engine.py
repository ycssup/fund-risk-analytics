import numpy as np
import pandas as pd
from typing import Dict, Optional


# These thresholds are intentionally grouped at the top of the file
# so you can tune them later without changing the signal logic itself.
SIGNAL_THRESHOLDS = {
    "volatility": {
        "low_max": 0.10,
        "medium_max": 0.25,
    },
    "drawdown": {
        "shallow_min": -0.10,
        "moderate_min": -0.25,
    },
    "sharpe": {
        "strong_min": 1.00,
        "average_min": 0.50,
    },
    "tail_risk": {
        "low_max": 0.03,
        "elevated_max": 0.08,
    },
    "rolling_trend": {
        # Compare the latest rolling volatility against a recent baseline.
        "improving_change_max": -0.10,
        "deteriorating_change_min": 0.10,
        "baseline_points": 3,
    },
}


def _get_metric(metric_categories: Dict[str, Dict[str, object]], category: str, metric: str):
    """
    Read one metric from the existing metric_categories dictionary.

    This helper keeps the rest of the signal code short and makes it easy
    to reuse the metrics structure that already exists in the project.
    """
    return metric_categories.get(category, {}).get(metric, np.nan)


def _coerce_float(value) -> float:
    """
    Convert a metric value to float when possible, otherwise return NaN.
    """
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def volatility_signal(annualized_volatility: float) -> str:
    """
    Convert annualized volatility into a simple absolute risk label.
    """
    annualized_volatility = _coerce_float(annualized_volatility)
    if np.isnan(annualized_volatility):
        return "medium"

    thresholds = SIGNAL_THRESHOLDS["volatility"]
    if annualized_volatility < thresholds["low_max"]:
        return "low"
    if annualized_volatility < thresholds["medium_max"]:
        return "medium"
    return "high"


def drawdown_signal(max_drawdown: float) -> str:
    """
    Convert maximum drawdown into a simple downside-depth label.
    """
    max_drawdown = _coerce_float(max_drawdown)
    if np.isnan(max_drawdown):
        return "moderate"

    thresholds = SIGNAL_THRESHOLDS["drawdown"]
    if max_drawdown >= thresholds["shallow_min"]:
        return "shallow"
    if max_drawdown >= thresholds["moderate_min"]:
        return "moderate"
    return "deep"


def sharpe_signal(sharpe_ratio: float) -> str:
    """
    Convert Sharpe ratio into a simple efficiency label.
    """
    sharpe_ratio = _coerce_float(sharpe_ratio)
    if np.isnan(sharpe_ratio):
        return "average"

    thresholds = SIGNAL_THRESHOLDS["sharpe"]
    if sharpe_ratio >= thresholds["strong_min"]:
        return "strong"
    if sharpe_ratio >= thresholds["average_min"]:
        return "average"
    return "weak"


def tail_risk_signal(tail_metrics: Dict[str, float]) -> str:
    """
    Convert existing tail-risk metrics into a single qualitative label.

    Assumption:
    - tail-risk metrics are stored as negative loss-style numbers
      (for example, -0.03 means a 3% loss)

    Rule:
    - pick the most severe tail loss using the minimum value
    - convert that loss to a positive magnitude with abs(...)
    - compare the positive magnitude against the configured thresholds

    This keeps the logic simple and avoids introducing a separate
    tail-risk calculation pipeline.
    """
    if not tail_metrics:
        return "elevated"

    available_values = [
        _coerce_float(value)
        for value in tail_metrics.values()
        if not np.isnan(_coerce_float(value))
    ]
    if not available_values:
        return "elevated"

    worst_tail_loss = min(available_values)
    tail_loss_abs = abs(worst_tail_loss)
    thresholds = SIGNAL_THRESHOLDS["tail_risk"]
    if tail_loss_abs < thresholds["low_max"]:
        return "low"
    if tail_loss_abs < thresholds["elevated_max"]:
        return "elevated"
    return "high"


def rolling_trend_signal(df: Optional[pd.DataFrame]) -> str:
    """
    Read existing rolling volatility columns and classify the current trend.

    Rule:
    - pick the longest rolling volatility window available
    - compare the latest valid observation with the average of a few recent
      previous points
    - lower rolling volatility -> improving
    - higher rolling volatility -> deteriorating
    """
    if df is None or df.empty:
        return "stable"

    rolling_vol_cols = [
        col for col in df.columns if col.startswith("rolling_vol_")
    ]
    if not rolling_vol_cols:
        return "stable"

    # Prefer the longest window because it is usually less noisy.
    selected_col = max(
        rolling_vol_cols,
        key=lambda col: int(col.split("_")[-1]) if col.split("_")[-1].isdigit() else -1,
    )
    rolling_series = pd.to_numeric(df[selected_col], errors="coerce").dropna()
    baseline_points = SIGNAL_THRESHOLDS["rolling_trend"]["baseline_points"]

    if len(rolling_series) < baseline_points + 1:
        return "stable"

    latest_value = rolling_series.iloc[-1]
    baseline = rolling_series.iloc[-(baseline_points + 1) : -1].mean()
    if baseline == 0 or np.isnan(baseline):
        return "stable"

    change_ratio = (latest_value - baseline) / baseline
    thresholds = SIGNAL_THRESHOLDS["rolling_trend"]

    if change_ratio <= thresholds["improving_change_max"]:
        return "improving"
    if change_ratio >= thresholds["deteriorating_change_min"]:
        return "deteriorating"
    return "stable"


def overall_risk_level(
    vol_signal: str,
    drawdown_signal_value: str,
    sharpe_signal_value: str,
    tail_signal: str,
) -> str:
    """
    Combine single signals into an overall risk level.

    The logic uses rule priority rather than a pure additive score:
    - first check hard-risk conditions that should dominate the judgment
    - then check broader moderate-risk conditions
    - otherwise classify the profile as Low risk
    """
    if drawdown_signal_value == "deep":
        return "High"

    if vol_signal == "high" and tail_signal == "high":
        return "High"

    if vol_signal == "high" and sharpe_signal_value == "weak":
        return "High"

    if (
        drawdown_signal_value == "moderate"
        or vol_signal == "medium"
        or tail_signal == "elevated"
        or sharpe_signal_value == "weak"
    ):
        return "Moderate"

    return "Low"


def key_risk_driver(
    vol_signal: str,
    drawdown_signal_value: str,
    tail_signal: str,
) -> str:
    """
    Identify the main absolute risk driver using simple priority rules.

    Interpretation logic:
    - deep drawdown takes priority because it usually dominates downside review
    - high volatility is the next strongest standalone driver
    - high tail risk is then checked as a separate dominant source of concern
    - if no single driver dominates, check whether multiple areas are elevated
    - otherwise return Balanced
    """
    if drawdown_signal_value == "deep":
        return "Drawdown"

    if vol_signal == "high":
        return "Volatility"

    if tail_signal == "high":
        return "Tail Risk"

    elevated_count = 0
    if vol_signal in {"medium", "high"}:
        elevated_count += 1
    if drawdown_signal_value in {"moderate", "deep"}:
        elevated_count += 1
    if tail_signal in {"elevated", "high"}:
        elevated_count += 1

    if elevated_count >= 2:
        return "Multiple Factors"

    return "Balanced"


def monitoring_flag(overall_risk: str, trend_signal: str) -> str:
    """
    Translate combined risk judgment into an operational monitoring flag.

    The logic is intentionally less aggressive than a direct trend-triggered
    escalation rule:
    - High overall risk -> Escalate
    - Moderate overall risk -> Watch
    - Low overall risk with a deteriorating trend -> Watch
    - otherwise -> Normal
    """
    if overall_risk == "High":
        return "Escalate"

    if overall_risk == "Moderate":
        return "Watch"

    if overall_risk == "Low" and trend_signal == "deteriorating":
        return "Watch"

    return "Normal"


def generate_risk_signals(
    metric_categories: Dict[str, Dict[str, object]],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Main entry point for the rule-based signal engine.

    Inputs:
    - metric_categories: existing project metrics dictionary
    - df: existing analysis DataFrame, used only for rolling-trend inspection

    Output:
    - a dictionary of qualitative absolute risk signals

    Flow:
    raw metrics -> single signals -> overall judgment
    """
    annualized_volatility = _get_metric(metric_categories, "Risk Metrics", "annualized_volatility")
    max_drawdown = _get_metric(metric_categories, "Risk Metrics", "max_drawdown")
    sharpe_ratio = _get_metric(metric_categories, "Risk-Adjusted Return", "sharpe_ratio")
    tail_metrics = metric_categories.get("Tail Risk Metrics", {})

    vol_signal_value = volatility_signal(annualized_volatility)
    drawdown_signal_value = drawdown_signal(max_drawdown)
    sharpe_signal_value = sharpe_signal(sharpe_ratio)
    tail_signal_value = tail_risk_signal(tail_metrics)
    trend_signal_value = rolling_trend_signal(df)

    overall_risk_value = overall_risk_level(
        vol_signal=vol_signal_value,
        drawdown_signal_value=drawdown_signal_value,
        sharpe_signal_value=sharpe_signal_value,
        tail_signal=tail_signal_value,
    )

    key_risk_driver_value = key_risk_driver(
        vol_signal=vol_signal_value,
        drawdown_signal_value=drawdown_signal_value,
        tail_signal=tail_signal_value,
    )

    monitoring_flag_value = monitoring_flag(
        overall_risk=overall_risk_value,
        trend_signal=trend_signal_value,
    )

    return {
        "vol_signal": vol_signal_value,
        "drawdown_signal": drawdown_signal_value,
        "sharpe_signal": sharpe_signal_value,
        "tail_signal": tail_signal_value,
        "trend_signal": trend_signal_value,
        "overall_risk": overall_risk_value,
        "key_risk_driver": key_risk_driver_value,
        "monitoring_flag": monitoring_flag_value,
    }
