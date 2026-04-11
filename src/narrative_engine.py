from typing import Dict, Optional

import pandas as pd


# These mappings convert structured risk signals into plain-English phrases.
# Keeping them in dictionaries makes the narrative deterministic and easy to edit.
SIGNAL_PHRASES = {
    "vol_signal": {
        "low": "low volatility",
        "medium": "moderate volatility",
        "high": "elevated volatility",
    },
    "drawdown_signal": {
        "shallow": "limited drawdown",
        "moderate": "a moderate level of drawdown",
        "deep": "significant drawdown pressure",
    },
    "sharpe_signal": {
        "strong": "strong risk-adjusted performance",
        "average": "moderate risk-adjusted performance",
        "weak": "weak risk-adjusted performance",
    },
    "trend_signal": {
        "improving": "risk conditions are improving",
        "deteriorating": "risk conditions are deteriorating",
        "stable": "risk conditions remain stable",
    },
    "key_risk_driver": {
        "Drawdown": "primarily driven by drawdown pressure",
        "Volatility": "primarily driven by elevated volatility",
        "Tail Risk": "primarily driven by tail-risk exposure",
        "Multiple Factors": "driven by multiple risk factors",
        "Balanced": "supported by a balanced risk profile",
    },
}


def _signal_text(signals: Dict[str, str], signal_name: str, default_text: str) -> str:
    """
    Safely map one signal value into narrative text.

    If a signal is missing or unexpected, we fall back to a neutral phrase
    so the report still produces a readable paragraph.
    """
    signal_value = signals.get(signal_name)
    return SIGNAL_PHRASES.get(signal_name, {}).get(signal_value, default_text)


def _format_pct(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_num(value) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{value:.4f}"


def generate_risk_narrative(
    signals: dict,
    benchmark_metrics: Optional[Dict[str, float]] = None,
    benchmark_name: str = "Benchmark",
) -> str:
    """
    Build a professional deterministic risk commentary from the risk signals.

    Input:
    - signals: dictionary produced by src.signal_engine.generate_risk_signals

    Output:
    - one structured paragraph that describes:
      1. volatility and drawdown
      2. risk-adjusted performance
      3. trend in risk conditions
      4. overall risk conclusion and monitoring status
    """
    vol_text = _signal_text(signals, "vol_signal", "an unclear volatility profile")
    drawdown_text = _signal_text(signals, "drawdown_signal", "an unclear drawdown profile")
    sharpe_text = _signal_text(signals, "sharpe_signal", "unclear risk-adjusted performance")
    trend_text = _signal_text(signals, "trend_signal", "risk conditions are mixed")
    driver_text = _signal_text(signals, "key_risk_driver", "supported by a balanced risk profile")

    overall_risk = signals.get("overall_risk", "Unknown")
    monitoring_flag = signals.get("monitoring_flag", "Normal")
    benchmark_metrics = benchmark_metrics or {}
    annualized_excess_return = benchmark_metrics.get("annualized_excess_return")
    tracking_error = benchmark_metrics.get("tracking_error")
    information_ratio = benchmark_metrics.get("information_ratio")
    cumulative_excess_return = benchmark_metrics.get("cumulative_excess_return")

    if pd.isna(annualized_excess_return):
        benchmark_sentence = (
            f"Benchmark-relative performance versus {benchmark_name} is not available from the current aligned sample."
        )
    else:
        relative_direction = "outperformed" if annualized_excess_return >= 0 else "underperformed"
        tracking_text = (
            "with tracking error not available"
            if pd.isna(tracking_error)
            else f"with tracking error of {_format_pct(tracking_error)}"
        )
        cumulative_text = (
            "Cumulative excess return is not available."
            if pd.isna(cumulative_excess_return)
            else f"Cumulative excess return over the aligned sample is {_format_pct(cumulative_excess_return)}."
        )
        benchmark_sentence = (
            f"The fund {relative_direction} {benchmark_name}, delivering annualized excess return "
            f"of {_format_pct(annualized_excess_return)} {tracking_text} and an information ratio of "
            f"{_format_num(information_ratio)}. This benchmark-relative profile indicates "
            f"{'effective active risk taking' if annualized_excess_return >= 0 and pd.notna(information_ratio) and information_ratio > 0 else 'that active risk has not yet translated into consistent excess return'}. "
            f"{cumulative_text}"
        )

    sentences = [
        f"The fund exhibits {vol_text}, with {drawdown_text}.",
        f"The fund shows {sharpe_text}, while {trend_text}.",
        benchmark_sentence,
        (
            f"Overall, the portfolio is assessed as {overall_risk} risk, "
            f"{driver_text}. "
            f"Monitoring status: {monitoring_flag}."
        ),
    ]

    return " ".join(sentences)
