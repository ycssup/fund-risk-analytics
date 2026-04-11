import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis_pipeline import run_analysis_pipeline


DEFAULT_INPUT_PATH = "data/sample_nav_data.xlsx"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_BENCHMARK_CANDIDATES = (
    "data/benchmark.xlsx",
    "data/benchmark.csv",
    "data/benchmark.xls",
)


def format_percentage(value) -> str:
    """
    Format decimal values as percentages for console output.
    """
    if pd.isna(value):
        return "NaN"

    return f"{value:.2%}"


def format_date(value) -> str:
    """
    Format timestamps for console output.
    """
    if pd.isna(value):
        return "NaN"

    return pd.to_datetime(value).strftime("%Y-%m-%d")


def format_return_table(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """
    Format a return table for readable console output.
    """
    display_table = df.copy()
    for column in display_table.columns:
        if "return" in column:
            display_table[column] = display_table[column].map(format_percentage)
    return display_table


def format_metric_label(metric_name: str) -> str:
    """
    Convert snake_case metric names into presentation-friendly labels.
    """
    label = metric_name.replace("_", " ").title()
    return label.replace("Cvar", "CVaR").replace("Var", "VaR").replace("Es", "ES")


def resolve_benchmark_path(benchmark_arg: str | None) -> str:
    """
    Resolve the benchmark file path from CLI input or default project locations.
    """
    if benchmark_arg:
        return benchmark_arg

    for candidate in DEFAULT_BENCHMARK_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    raise ValueError(
        "Benchmark input is required for the aligned analysis pipeline. "
        "Pass --benchmark or add data/benchmark.xlsx, data/benchmark.csv, or data/benchmark.xls."
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the analysis runner.
    """
    parser = argparse.ArgumentParser(description="Run benchmark-aligned fund risk analysis.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Fund NAV file path. Supports CSV, XLSX, and XLS. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark level file path. The second column must be a benchmark level series, not returns.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory root. Charts will be saved under <output>/charts and reports under <output>/reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Optional friendly benchmark name to display in charts, tables, and console output.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the benchmark-aware CLI analysis flow and print key outputs.
    """
    args = parse_args()
    benchmark_path = resolve_benchmark_path(args.benchmark)

    print(f"Loading fund data: {args.input}")
    print(f"Loading benchmark data: {benchmark_path}")
    print("Benchmark assumption: the second benchmark column must be a level series, not precomputed returns.")
    print("Alignment rule: fund NAV dates define the analysis timeline, and benchmark levels are mapped backward onto those dates.")

    analysis = run_analysis_pipeline(
        fund_input_path=args.input,
        benchmark_input_path=benchmark_path,
        output_dir=args.output,
        benchmark_name=args.benchmark_name,
    )

    df = analysis["df"]
    return_metrics = analysis["metric_categories"]["Return Metrics"]
    risk_metrics = analysis["metric_categories"]["Risk Metrics"]
    benchmark_metrics = analysis["metric_categories"]["Benchmark Comparison"]
    tail_metrics = analysis["metric_categories"]["Tail Risk Metrics"]
    risk_adjusted_metrics = analysis["metric_categories"]["Risk-Adjusted Return"]
    metrics_summary_table = analysis["metrics_summary_table"]
    alignment_metadata = analysis["alignment_metadata"]
    benchmark_name = analysis["benchmark_name"]

    print(f"Benchmark name used: {benchmark_name}")

    print(f"Inferred fund frequency: {alignment_metadata['fund_frequency']}")
    print(f"Inferred benchmark frequency: {alignment_metadata['benchmark_frequency']}")
    print(f"Overlapping observations after alignment: {alignment_metadata['overlapping_observations']}")
    print(f"Fund rows dropped before benchmark history starts: {alignment_metadata['dropped_unmatched_fund_rows']}")
    print(f"Alignment method: {alignment_metadata['alignment_method']}")
    print(
        "Aligned date range: "
        f"{format_date(alignment_metadata['aligned_start_date'])} to "
        f"{format_date(alignment_metadata['aligned_end_date'])}"
    )
    print(f"Annualization factor: {analysis['periods_per_year']:.0f}")
    print(f"Rolling windows: {analysis['rolling_windows']}")

    print("\n=== Aligned Dataset Preview ===")
    display_df = df.copy()
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")

    preview_numeric_cols = [
        col
        for col in [
            "nav",
            "benchmark",
            "fund_return",
            "benchmark_return",
            "period_excess_return",
            "cumulative_return",
            "benchmark_cumulative_return",
            "cumulative_excess_return",
        ]
        if col in display_df.columns
    ]
    display_df[preview_numeric_cols] = display_df[preview_numeric_cols].round(6)
    for return_col in [
        "fund_return",
        "benchmark_return",
        "period_excess_return",
        "cumulative_return",
        "benchmark_cumulative_return",
        "cumulative_excess_return",
    ]:
        if return_col in display_df.columns:
            display_df[return_col] = df[return_col].map(format_percentage)
    print(display_df.head(12).to_string(index=False))

    print("\n=== Return Metrics ===")
    for metric_name, metric_value in return_metrics.items():
        if "months" in metric_name:
            print(f"{format_metric_label(metric_name)}: {metric_value:.0f}")
        elif metric_name in {"return_skewness"}:
            print(
                f"{format_metric_label(metric_name)}: {metric_value:.4f}"
                if pd.notna(metric_value)
                else f"{format_metric_label(metric_name)}: NaN"
            )
        else:
            print(f"{format_metric_label(metric_name)}: {format_percentage(metric_value)}")

    print(f"\n=== Benchmark Comparison vs {benchmark_name} ===")
    for metric_name, metric_value in benchmark_metrics.items():
        if metric_name == "information_ratio":
            print(
                f"{format_metric_label(metric_name)}: {metric_value:.4f}"
                if pd.notna(metric_value)
                else f"{format_metric_label(metric_name)}: NaN"
            )
        else:
            print(f"{format_metric_label(metric_name)}: {format_percentage(metric_value)}")

    print("\n=== Monthly Returns ===")
    print(format_return_table(analysis["monthly_return_table"], "monthly_return").to_string(index=False))
    print(f"\n=== Monthly Returns: {benchmark_name} ===")
    print(
        format_return_table(
            analysis["benchmark_monthly_return_table"],
            "benchmark_monthly_return",
        ).to_string(index=False)
    )
    print(f"\n=== Monthly Excess Returns vs {benchmark_name} ===")
    print(
        format_return_table(
            analysis["monthly_excess_return_table"][
                ["month", "fund_monthly_return", "benchmark_monthly_return", "monthly_excess_return"]
            ].copy(),
            "monthly_excess_return",
        ).to_string(index=False)
    )
    print("\n=== Annual Returns ===")
    print(format_return_table(analysis["annual_return_table"], "annual_return").to_string(index=False))
    print(f"\n=== Annual Returns: {benchmark_name} ===")
    print(
        format_return_table(
            analysis["benchmark_annual_return_table"],
            "benchmark_annual_return",
        ).to_string(index=False)
    )
    print(f"\n=== Annual Excess Returns vs {benchmark_name} ===")
    print(
        format_return_table(
            analysis["annual_excess_return_table"][
                ["year", "fund_annual_return", "benchmark_annual_return", "annual_excess_return"]
            ].copy(),
            "annual_excess_return",
        ).to_string(index=False)
    )

    print("\n=== Risk Metrics ===")
    print(f"Annualized Return: {format_percentage(risk_metrics['annualized_return'])}")
    print(f"Annualized Volatility: {format_percentage(risk_metrics['annualized_volatility'])}")
    print(f"Maximum Drawdown: {format_percentage(risk_metrics['max_drawdown'])}")
    print(f"Maximum Drawdown Date: {format_date(risk_metrics['max_drawdown_date'])}")
    recovery_days = risk_metrics["max_drawdown_recovery_days"]
    if pd.notna(recovery_days):
        print(f"Maximum Drawdown Recovery Days: {recovery_days:.0f}")
    else:
        print("Maximum Drawdown Recovery Days: Not recovered")

    print("\n=== Tail Risk Metrics ===")
    for metric_name, metric_value in tail_metrics.items():
        print(f"{format_metric_label(metric_name)}: {format_percentage(metric_value)}")

    print("\n=== Risk-Adjusted Return Metrics ===")
    for metric_name, metric_value in risk_adjusted_metrics.items():
        if pd.isna(metric_value):
            print(f"{format_metric_label(metric_name)}: NaN")
        else:
            print(f"{format_metric_label(metric_name)}: {metric_value:.4f}")

    print("\n=== Risk Signals ===")
    for signal_name, signal_value in analysis["risk_signals"].items():
        print(f"{format_metric_label(signal_name)}: {signal_value}")

    print("\n=== Risk Narrative ===")
    print(analysis["risk_narrative"])

    print("\n=== Drawdown Frequency ===")
    for frequency_name, frequency_table in analysis["drawdown_frequencies"].items():
        print(f"\n{frequency_name.title()}")
        print(frequency_table.to_string(index=False))

    print("\n=== Metrics Summary Table ===")
    print(metrics_summary_table.to_string(index=False))

    print("\nAnalysis completed successfully.")
    print(f"Charts saved under: {os.path.join(args.output, 'charts')}")
    print(f"Reports saved under: {os.path.join(args.output, 'reports')}")


if __name__ == "__main__":
    main()
