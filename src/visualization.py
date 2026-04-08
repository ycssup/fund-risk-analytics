import os
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

CHART_DIR = "output/charts"
REPORT_DIR = "output/reports"


def _ensure_output_dirs() -> None:
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def _date_axis(df: pd.DataFrame):
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")

    return df.index


def _format_metric_value(value, metric_name: str, percentage_metrics: Iterable[str]) -> str:
    if pd.isna(value):
        return "NaN"

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if metric_name == "annualization_factor":
        return f"{value:.0f}"

    if "months" in metric_name or "days" in metric_name:
        return f"{value:.0f}"

    if metric_name in percentage_metrics:
        return f"{value:.2%}"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}"

    return str(value)


def build_metrics_summary_table(
    metric_categories: Dict[str, Dict[str, object]],
    percentage_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build a categorized metrics summary table.
    """
    percentage_metrics = set(percentage_metrics or [])
    rows = []

    for category, metrics in metric_categories.items():
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "category": category,
                    "metric": metric_name,
                    "value": _format_metric_value(
                        metric_value,
                        metric_name,
                        percentage_metrics,
                    ),
                }
            )

    return pd.DataFrame(rows, columns=["category", "metric", "value"])


def save_metrics_summary_table(
    metric_categories: Dict[str, Dict[str, object]],
    percentage_metrics: Optional[Iterable[str]] = None,
    output_path: str = f"{CHART_DIR}/metrics_summary_table.png",
) -> pd.DataFrame:
    """
    Save a categorized metrics summary table as PNG and CSV.
    """
    _ensure_output_dirs()
    table_df = build_metrics_summary_table(
        metric_categories,
        percentage_metrics=percentage_metrics,
    )

    csv_path = os.path.join(REPORT_DIR, "metrics_summary_table.csv")
    table_df.to_csv(csv_path, index=False)

    fig_height = max(4, 0.35 * len(table_df) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=["Category", "Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    return table_df


def plot_nav_and_drawdown(df: pd.DataFrame, output_path: str = f"{CHART_DIR}/nav_drawdown.png") -> None:
    """
    Plot NAV as a line chart and drawdown as an area/line chart.
    """
    _ensure_output_dirs()
    x_axis = _date_axis(df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x_axis, df["nav"], label="NAV", color="#1f77b4", linewidth=1.8)
    axes[0].set_title("NAV Curve")
    axes[0].set_ylabel("NAV")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_axis, df["drawdown"], label="Drawdown", color="#d62728", linewidth=1.5)
    axes[1].fill_between(x_axis, df["drawdown"], 0, color="#d62728", alpha=0.25)
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_metrics(df: pd.DataFrame, output_path: str = f"{CHART_DIR}/rolling_metrics.png") -> None:
    """
    Plot rolling volatility and rolling Sharpe ratio as line charts.
    """
    _ensure_output_dirs()
    x_axis = _date_axis(df)
    rolling_vol_cols = [col for col in df.columns if col.startswith("rolling_vol_")]
    rolling_sharpe_cols = [col for col in df.columns if col.startswith("rolling_sharpe_")]

    if not rolling_vol_cols and "rolling_vol" in df.columns:
        rolling_vol_cols = ["rolling_vol"]

    if not rolling_sharpe_cols and "rolling_sharpe" in df.columns:
        rolling_sharpe_cols = ["rolling_sharpe"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for col in rolling_vol_cols:
        label = col.replace("rolling_vol_", "Vol ")
        axes[0].plot(x_axis, df[col], label=label)
    axes[0].set_title("Rolling Volatility")
    axes[0].set_ylabel("Volatility")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for col in rolling_sharpe_cols:
        label = col.replace("rolling_sharpe_", "Sharpe ")
        axes[1].plot(x_axis, df[col], label=label)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Rolling Sharpe")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Sharpe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_monthly_returns(
    monthly_return_table: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/monthly_returns.png",
) -> None:
    """
    Plot monthly returns as a bar chart.
    """
    _ensure_output_dirs()
    if monthly_return_table.empty:
        return

    data = monthly_return_table.copy()
    colors = np.where(data["monthly_return"] >= 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(data["month"], data["monthly_return"], color=colors)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("Monthly Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_annual_returns(
    annual_return_table: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/annual_returns.png",
) -> None:
    """
    Plot annual returns as a bar chart.
    """
    _ensure_output_dirs()
    if annual_return_table.empty:
        return

    data = annual_return_table.copy()
    data["year"] = data["year"].astype(str)
    colors = np.where(data["annual_return"] >= 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["year"], data["annual_return"], color=colors)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("Annual Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown_frequency(
    drawdown_frequencies: Dict[str, pd.DataFrame],
    output_path: str = f"{CHART_DIR}/drawdown_frequency.png",
) -> None:
    """
    Plot drawdown frequency tables as bar charts.
    """
    _ensure_output_dirs()
    if not drawdown_frequencies:
        return

    fig, axes = plt.subplots(len(drawdown_frequencies), 1, figsize=(14, 4 * len(drawdown_frequencies)))
    if len(drawdown_frequencies) == 1:
        axes = [axes]

    for ax, (frequency_name, frequency_table) in zip(axes, drawdown_frequencies.items()):
        if frequency_table.empty:
            ax.set_title(f"{frequency_name.title()} Drawdown Frequency")
            ax.axis("off")
            continue

        data = frequency_table.copy()
        ax.bar(
            data["period"],
            data["drawdown_observation_frequency"],
            color="#9467bd",
            alpha=0.85,
            label="Observation Frequency",
        )
        ax2 = ax.twinx()
        ax2.plot(
            data["period"],
            data["drawdown_event_count"],
            color="#ff7f0e",
            marker="o",
            label="Event Count",
        )

        ax.set_title(f"{frequency_name.title()} Drawdown Frequency")
        ax.set_ylabel("Observation Frequency")
        ax2.set_ylabel("Event Count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_analysis_visualizations(
    df: pd.DataFrame,
    metric_categories: Dict[str, Dict[str, object]],
    monthly_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    drawdown_frequencies: Dict[str, pd.DataFrame],
    percentage_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Generate all analysis tables and charts.
    """
    metrics_table = save_metrics_summary_table(
        metric_categories,
        percentage_metrics=percentage_metrics,
    )
    plot_nav_and_drawdown(df)
    plot_rolling_metrics(df)
    plot_monthly_returns(monthly_return_table)
    plot_annual_returns(annual_return_table)
    plot_drawdown_frequency(drawdown_frequencies)

    return metrics_table
