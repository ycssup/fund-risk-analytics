from typing import Dict, Optional

import pandas as pd

from src.data_loader import load_and_align_datasets
from src.drawdown_analysis import calculate_drawdown, drawdown_frequency_summary, max_drawdown_details
from src.narrative_engine import generate_risk_narrative
from src.relative_performance import (
    build_relative_period_table,
    calculate_annualized_excess_return,
    calculate_cumulative_excess_returns,
)
from src.return_metrics import return_summary_metrics
from src.risk_adjusted_return import benchmark_comparison_metrics, risk_adjusted_return_metrics
from src.risk_metrics import annualized_return, annualized_volatility, tail_risk_metrics
from src.rolling_metrics import add_rolling_metrics
from src.signal_engine import generate_risk_signals
from src.visualization import generate_analysis_visualizations


def percentage_metric_names(tail_metrics: Dict[str, float]) -> set:
    """
    Return metric names that should be formatted as percentages in tables.
    """
    return {
        "inception_annualized_return",
        "inception_return",
        "year_to_date_return",
        "one_year_return",
        "win_rate",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "benchmark_annualized_return",
        "benchmark_cumulative_return",
        "annualized_excess_return",
        "cumulative_excess_return",
        "period_excess_hit_rate",
        "tracking_error",
        *tail_metrics.keys(),
    }


def build_metric_categories(
    df: pd.DataFrame,
    fund_input_path: str,
    benchmark_input_path: str,
    benchmark_name: str,
    frequency: str,
    periods_per_year: float,
    rolling_windows,
    return_metrics: Dict[str, float],
    ann_ret: float,
    ann_vol: float,
    drawdown_details: Dict[str, object],
    tail_metrics: Dict[str, float],
    risk_adjusted_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    """
    Build grouped summary-table categories for downstream reporting.
    """
    return {
        "Data Profile": {
            "fund_input_file": fund_input_path,
            "benchmark_input_file": benchmark_input_path,
            "benchmark_name": benchmark_name,
            "start_date": df["date"].min(),
            "end_date": df["date"].max(),
            "frequency": frequency,
            "annualization_factor": periods_per_year,
            "observations": len(df),
            "rolling_windows": ", ".join(str(window) for window in rolling_windows),
        },
        "Return Metrics": return_metrics,
        "Risk Metrics": {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            **drawdown_details,
        },
        "Benchmark Comparison": benchmark_metrics,
        "Tail Risk Metrics": tail_metrics,
        "Risk-Adjusted Return": risk_adjusted_metrics,
    }


def run_analysis_pipeline(
    fund_input_path: str,
    benchmark_input_path: str,
    output_dir: Optional[str] = None,
    benchmark_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run the full benchmark-aware analysis pipeline on aligned fund and benchmark data.
    """
    df, alignment_metadata = load_and_align_datasets(
        fund_file_path=fund_input_path,
        benchmark_file_path=benchmark_input_path,
        benchmark_name=benchmark_name,
    )

    frequency = alignment_metadata["frequency"]
    periods_per_year = alignment_metadata["periods_per_year"]
    resolved_benchmark_name = alignment_metadata["benchmark_name"]

    cumulative_fund_return, cumulative_benchmark_return, cumulative_excess_return = calculate_cumulative_excess_returns(
        df["fund_return"],
        df["benchmark_return"],
    )
    df["cumulative_return"] = cumulative_fund_return.reindex(df.index)
    df["benchmark_cumulative_return"] = cumulative_benchmark_return.reindex(df.index)
    df["cumulative_excess_return"] = cumulative_excess_return.reindex(df.index)

    return_metrics = return_summary_metrics(df)
    monthly_relative_return_table = build_relative_period_table(
        df=df,
        freq="M",
        fund_level_col="nav",
        benchmark_level_col="benchmark",
        fund_output_col="fund_monthly_return",
        benchmark_output_col="benchmark_monthly_return",
        excess_output_col="monthly_excess_return",
    )
    annual_relative_return_table = build_relative_period_table(
        df=df,
        freq="Y",
        fund_level_col="nav",
        benchmark_level_col="benchmark",
        fund_output_col="fund_annual_return",
        benchmark_output_col="benchmark_annual_return",
        excess_output_col="annual_excess_return",
    )

    monthly_return_table = monthly_relative_return_table[["month", "fund_monthly_return"]].rename(
        columns={"fund_monthly_return": "monthly_return"}
    )
    benchmark_monthly_return_table = monthly_relative_return_table[["month", "benchmark_monthly_return"]].copy()
    monthly_excess_return_table = monthly_relative_return_table.copy()
    annual_return_table = annual_relative_return_table[["year", "fund_annual_return"]].rename(
        columns={"fund_annual_return": "annual_return"}
    )
    benchmark_annual_return_table = annual_relative_return_table[["year", "benchmark_annual_return"]].copy()
    annual_excess_return_table = annual_relative_return_table.copy()

    df, rolling_windows = add_rolling_metrics(df, freq=frequency)

    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(df, periods_per_year=periods_per_year)
    tail_metrics = tail_risk_metrics(df, periods_per_year=periods_per_year)
    annualized_excess_return = calculate_annualized_excess_return(
        df["fund_return"],
        df["benchmark_return"],
        periods_per_year=periods_per_year,
    )
    risk_adjusted_metrics = risk_adjusted_return_metrics(
        df,
        periods_per_year=periods_per_year,
    )
    benchmark_metrics = benchmark_comparison_metrics(
        df,
        periods_per_year=periods_per_year,
        annualized_excess_return=annualized_excess_return,
    )

    df = calculate_drawdown(df)
    drawdown_details = max_drawdown_details(df)
    drawdown_frequencies = drawdown_frequency_summary(df)

    percentage_metrics = percentage_metric_names(tail_metrics)
    metric_categories = build_metric_categories(
        df=df,
        fund_input_path=fund_input_path,
        benchmark_input_path=benchmark_input_path,
        benchmark_name=resolved_benchmark_name,
        frequency=frequency,
        periods_per_year=periods_per_year,
        rolling_windows=rolling_windows,
        return_metrics=return_metrics,
        ann_ret=ann_ret,
        ann_vol=ann_vol,
        drawdown_details=drawdown_details,
        tail_metrics=tail_metrics,
        risk_adjusted_metrics=risk_adjusted_metrics,
        benchmark_metrics=benchmark_metrics,
    )

    risk_signals = generate_risk_signals(metric_categories=metric_categories, df=df)
    metric_categories["Risk Signals"] = risk_signals
    risk_narrative = generate_risk_narrative(
        risk_signals,
        benchmark_metrics=benchmark_metrics,
        benchmark_name=resolved_benchmark_name,
    )
    metric_categories["Risk Narrative"] = {"risk_narrative": risk_narrative}

    metrics_summary_table = generate_analysis_visualizations(
        df=df,
        metric_categories=metric_categories,
        monthly_return_table=monthly_return_table,
        benchmark_monthly_return_table=benchmark_monthly_return_table,
        monthly_excess_return_table=monthly_excess_return_table,
        annual_return_table=annual_return_table,
        benchmark_annual_return_table=benchmark_annual_return_table,
        annual_excess_return_table=annual_excess_return_table,
        drawdown_frequencies=drawdown_frequencies,
        percentage_metrics=percentage_metrics,
        output_dir=output_dir,
        benchmark_name=resolved_benchmark_name,
    )

    return {
        "df": df,
        "alignment_metadata": alignment_metadata,
        "metric_categories": metric_categories,
        "metrics_summary_table": metrics_summary_table,
        "percentage_metrics": percentage_metrics,
        "monthly_return_table": monthly_return_table,
        "benchmark_monthly_return_table": benchmark_monthly_return_table,
        "monthly_excess_return_table": monthly_excess_return_table,
        "annual_return_table": annual_return_table,
        "benchmark_annual_return_table": benchmark_annual_return_table,
        "annual_excess_return_table": annual_excess_return_table,
        "drawdown_frequencies": drawdown_frequencies,
        "data_frequency": frequency,
        "periods_per_year": periods_per_year,
        "rolling_windows": rolling_windows,
        "risk_signals": risk_signals,
        "risk_narrative": risk_narrative,
        "benchmark_name": resolved_benchmark_name,
    }
