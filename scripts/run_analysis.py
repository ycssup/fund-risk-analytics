import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_nav_data
from src.return_metrics import (
    calculate_cumulative_returns,
    calculate_daily_returns,
    return_summary_metrics,
    return_tables,
)
from src.frequency import infer_data_frequency
from src.risk_metrics import annualized_return, annualized_volatility, tail_risk_metrics
from src.risk_adjusted_return import risk_adjusted_return_metrics
from src.drawdown_analysis import calculate_drawdown, drawdown_frequency_summary, max_drawdown_details
from src.rolling_metrics import add_rolling_metrics
from src.visualization import generate_analysis_visualizations


def format_percentage(value) -> str:
    if pd.isna(value):
        return "NaN"

    return f"{value:.2%}"


def format_date(value) -> str:
    if pd.isna(value):
        return "NaN"

    return pd.to_datetime(value).strftime("%Y-%m-%d")


def format_return_table(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    display_table = df.copy()
    display_table[return_col] = display_table[return_col].map(format_percentage)

    return display_table


if __name__ == "__main__":
    file_path = "data/nav_sanitized_low_corr.xlsx"

    # Step 1: load NAV data
    df = load_nav_data(file_path)

    # Step 2: determine data frequency before any return or risk analysis
    data_frequency, periods_per_year = infer_data_frequency(df, strict=True)

    print(df.head())
    print(df.info())
    print(f"Detected Data Frequency: {data_frequency}")
    print(f"Annualization Factor: {periods_per_year:.0f}")

    # Step 3: calculate returns
    df = calculate_daily_returns(df)
    df = calculate_cumulative_returns(df)
    return_metrics = return_summary_metrics(df)
    monthly_return_table, annual_return_table = return_tables(df)

    # Step 4: rolling metrics
    df, rolling_windows = add_rolling_metrics(df, freq=data_frequency)
    print(f"Rolling Windows: {rolling_windows}")

    # Step 5: calculate risk metrics
    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(df, periods_per_year=periods_per_year)
    tail_metrics = tail_risk_metrics(df, periods_per_year=periods_per_year)
    risk_adjusted_metrics = risk_adjusted_return_metrics(df, periods_per_year=periods_per_year)

    # Step 6: calculate drawdown
    df = calculate_drawdown(df)
    drawdown_details = max_drawdown_details(df)
    drawdown_frequencies = drawdown_frequency_summary(df)

    # Step 7: visualization
    percentage_return_metrics = {
        "inception_annualized_return",
        "inception_return",
        "year_to_date_return",
        "one_year_return",
        "win_rate",
    }
    percentage_risk_metrics = {
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    }
    percentage_tail_metrics = set(tail_metrics.keys())
    percentage_metrics = (
        percentage_return_metrics
        | percentage_risk_metrics
        | percentage_tail_metrics
    )
    metric_categories = {
        "Data Profile": {
            "data_frequency": data_frequency,
            "annualization_factor": periods_per_year,
        },
        "Return Metrics": return_metrics,
        "Risk Metrics": {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            **drawdown_details,
        },
        "Tail Risk Metrics": tail_metrics,
        "Risk-Adjusted Return Metrics": risk_adjusted_metrics,
    }
    metrics_summary_table = generate_analysis_visualizations(
        df=df,
        metric_categories=metric_categories,
        monthly_return_table=monthly_return_table,
        annual_return_table=annual_return_table,
        drawdown_frequencies=drawdown_frequencies,
        percentage_metrics=percentage_metrics,
    )

    print("=== NAV Data with Returns ===")
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(6)
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    for return_col in ["daily_return", "cumulative_return"]:
        if return_col in display_df.columns:
            display_df[return_col] = df[return_col].map(format_percentage)
    print(display_df)
    print("\n=== Return Metrics ===")
    for metric_name, metric_value in return_metrics.items():
        if pd.isna(metric_value):
            print(f"{metric_name}: NaN")
        elif "months" in metric_name:
            print(f"{metric_name}: {metric_value}")
        elif metric_name in percentage_return_metrics:
            print(f"{metric_name}: {format_percentage(metric_value)}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    print("\n=== Monthly Returns ===")
    print(format_return_table(monthly_return_table, "monthly_return").to_string(index=False))
    print("\n=== Annual Returns ===")
    print(format_return_table(annual_return_table, "annual_return").to_string(index=False))
    print("\n=== Risk Metrics ===")
    print(f"Annualized Return: {format_percentage(ann_ret)}")
    print(f"Annualized Volatility: {format_percentage(ann_vol)}")
    print(f"Maximum Drawdown: {format_percentage(drawdown_details['max_drawdown'])}")
    print(f"Maximum Drawdown Date: {format_date(drawdown_details['max_drawdown_date'])}")
    recovery_days = drawdown_details["max_drawdown_recovery_days"]
    if recovery_days == recovery_days:
        print(f"Maximum Drawdown Recovery Days: {recovery_days:.0f}")
    else:
        print("Maximum Drawdown Recovery Days: Not recovered")
    print("\n=== Tail Risk Metrics ===")
    for metric_name, metric_value in tail_metrics.items():
        print(f"{metric_name}: {format_percentage(metric_value)}")
    print("\n=== Risk-Adjusted Return Metrics ===")
    for metric_name, metric_value in risk_adjusted_metrics.items():
        if pd.isna(metric_value):
            print(f"{metric_name}: NaN")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    print("\n=== Drawdown Frequency ===")
    for frequency_name, frequency_table in drawdown_frequencies.items():
        print(f"\n{frequency_name.title()}")
        print(frequency_table.to_string(index=False))
    print("\n=== Metrics Summary Table ===")
    print(metrics_summary_table.to_string(index=False))
