from typing import Dict


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


def generate_risk_narrative(signals: dict) -> str:
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

    sentences = [
        f"The fund exhibits {vol_text}, with {drawdown_text}.",
        f"The fund shows {sharpe_text}, while {trend_text}.",
        (
            f"Overall, the portfolio is assessed as {overall_risk} risk, "
            f"{driver_text}. "
            f"Monitoring status: {monitoring_flag}."
        ),
    ]

    return " ".join(sentences)
