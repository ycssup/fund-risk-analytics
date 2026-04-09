import argparse
import os
import sys
import textwrap
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_nav_data
from src.drawdown_analysis import calculate_drawdown, drawdown_frequency_summary, max_drawdown_details
from src.frequency import infer_data_frequency
from src.return_metrics import (
    calculate_cumulative_returns,
    calculate_daily_returns,
    return_summary_metrics,
    return_tables,
)
from src.risk_adjusted_return import risk_adjusted_return_metrics
from src.risk_metrics import annualized_return, annualized_volatility, tail_risk_metrics
from src.rolling_metrics import add_rolling_metrics
from src.signal_engine import generate_risk_signals
from src.narrative_engine import generate_risk_narrative
from src.visualization import CHART_DIR, REPORT_DIR, generate_analysis_visualizations


DEFAULT_INPUT_PATH = "data/sample_nav_data.xlsx"
DEFAULT_OUTPUT_PATH = f"{REPORT_DIR}/fund_risk_report.pdf"


def _percentage_metric_names(tail_metrics: Dict[str, float]) -> set:
    return {
        "inception_annualized_return",
        "inception_return",
        "year_to_date_return",
        "one_year_return",
        "win_rate",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        *tail_metrics.keys(),
    }


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


def _flatten_metric_categories(
    metric_categories: Dict[str, Dict[str, object]],
    percentage_metrics: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for category, metrics in metric_categories.items():
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "Category": category,
                    "Metric": metric_name,
                    "Value": _format_metric_value(metric_value, metric_name, percentage_metrics),
                }
            )

    return pd.DataFrame(rows, columns=["Category", "Metric", "Value"])


def _fmt_pct(value) -> str:
    if pd.isna(value):
        return "NaN"

    return f"{value:.2%}"


def _fmt_num(value) -> str:
    if pd.isna(value):
        return "NaN"

    return f"{value:.4f}"


def _fmt_date(value) -> str:
    if pd.isna(value):
        return "NaN"

    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _describe_level(value, metric_type: str) -> str:
    if pd.isna(value):
        return "not available"

    if metric_type == "volatility":
        if value < 0.10:
            return "low"
        if value < 0.25:
            return "moderate"
        return "elevated"

    if metric_type == "drawdown":
        if value > -0.10:
            return "limited"
        if value > -0.25:
            return "moderate"
        return "material"

    if metric_type == "sharpe":
        if value >= 1:
            return "strong"
        if value >= 0.5:
            return "acceptable"
        return "weak"

    return "available"


def generate_report_commentary(analysis: dict) -> Dict[str, List[str]]:
    """
    Convert calculated metrics into natural-language report commentary.
    """
    metrics = analysis["metric_categories"]
    profile = metrics["Data Profile"]
    returns = metrics["Return Metrics"]
    risks = metrics["Risk Metrics"]
    tail = metrics["Tail Risk Metrics"]
    adj = metrics["Risk-Adjusted Return Metrics"]

    data_frequency = profile.get("data_frequency", "unknown")
    annualization_factor = profile.get("annualization_factor", np.nan)
    rolling_windows = profile.get("rolling_windows", "not available")
    start_date = _fmt_date(profile.get("start_date", pd.NaT))
    end_date = _fmt_date(profile.get("end_date", pd.NaT))
    observations = profile.get("observations", "NaN")

    inception_return = returns.get("inception_return", np.nan)
    inception_annualized = returns.get("inception_annualized_return", np.nan)
    ytd_return = returns.get("year_to_date_return", np.nan)
    one_year_return = returns.get("one_year_return", np.nan)
    win_rate = returns.get("win_rate", np.nan)
    skewness = returns.get("return_skewness", np.nan)

    ann_vol = risks.get("annualized_volatility", np.nan)
    max_dd = risks.get("max_drawdown", np.nan)
    max_dd_date = risks.get("max_drawdown_date", pd.NaT)
    recovery_days = risks.get("max_drawdown_recovery_days", np.nan)

    sharpe = adj.get("sharpe_ratio", np.nan)
    sortino = adj.get("sortino_ratio", np.nan)
    calmar = adj.get("calmar_ratio", np.nan)
    treynor = adj.get("treynor_ratio", np.nan)
    treynor_black = adj.get("treynor_black_ratio", np.nan)

    recovery_text = (
        "not recovered within the available sample"
        if pd.isna(recovery_days)
        else f"recovered after {recovery_days:.0f} days"
    )

    commentary = {
        "Data and Frequency Assessment": [
            (
                f"The report is based on {observations} NAV observations from {start_date} "
                f"to {end_date}. The detected data frequency is {data_frequency}, so the "
                f"analysis uses an annualization factor of {annualization_factor:.0f}."
            ),
            (
                f"Rolling metrics are calculated with frequency-aware windows: {rolling_windows}. "
                "This prevents daily assumptions from being applied to weekly or monthly data."
            ),
        ],
        "Return Performance Commentary": [
            (
                f"Since inception, the fund generated a total return of {_fmt_pct(inception_return)}, "
                f"equivalent to an annualized return of {_fmt_pct(inception_annualized)} over the "
                "observed period."
            ),
            (
                f"The year-to-date return is {_fmt_pct(ytd_return)}, while the trailing one-year "
                f"return is {_fmt_pct(one_year_return)}. The win rate is {_fmt_pct(win_rate)}, "
                "which measures the share of positive periodic return observations."
            ),
            (
                f"The return distribution skewness is {_fmt_num(skewness)}. Positive skewness "
                "suggests relatively more upside tail observations, while negative skewness would "
                "indicate more downside tail behavior."
            ),
        ],
        "Risk and Drawdown Commentary": [
            (
                f"Annualized volatility is {_fmt_pct(ann_vol)}, which is classified as "
                f"{_describe_level(ann_vol, 'volatility')} based on the rule-of-thumb thresholds "
                "used in this report."
            ),
            (
                f"The maximum drawdown is {_fmt_pct(max_dd)}, occurring on {_fmt_date(max_dd_date)}. "
                f"This drawdown is classified as {_describe_level(max_dd, 'drawdown')} and has "
                f"{recovery_text}."
            ),
            (
                "Drawdown analysis should be read together with the NAV chart and drawdown frequency "
                "chart, because maximum drawdown captures only the deepest loss while frequency "
                "captures how persistently the fund stayed below prior highs."
            ),
        ],
        "Tail Risk Commentary": [
            (
                f"The 95% 1-day equivalent VaR (scaled from {data_frequency} returns) is {_fmt_pct(tail.get('var_95_1d_scaled', np.nan))}, "
                f"and the corresponding 10-day VaR is {_fmt_pct(tail.get('var_95_10d', np.nan))}. "
                "These figures estimate downside loss under historical simulation assumptions."
            ),
            (
                f"The 99% 1-day equivalent VaR (scaled from {data_frequency} returns) is {_fmt_pct(tail.get('var_99_1d_scaled', np.nan))}, "
                f"and the corresponding 10-day VaR is {_fmt_pct(tail.get('var_99_10d', np.nan))}. "
                "The 99% confidence level focuses on more severe tail observations."
            ),
            (
                f"The 95% 1-day equivalent CVaR/ES (scaled from {data_frequency} returns) is {_fmt_pct(tail.get('cvar_es_95_1d_scaled', np.nan))}, "
                f"while the 99% 1-day equivalent CVaR/ES (scaled from {data_frequency} returns) is {_fmt_pct(tail.get('cvar_es_99_1d_scaled', np.nan))}. "
                "CVaR/ES estimates the average loss within the tail, so it is useful for "
                "understanding loss severity beyond the VaR threshold."
            ),
            (
                "Note: For non-daily data, short-horizon VaR and CVaR are scaled estimates based on "
                "the underlying return frequency."
            ),
        ],
        "Risk-Adjusted Return Commentary": [
            (
                f"The Sharpe ratio is {_fmt_num(sharpe)}, which is classified as "
                f"{_describe_level(sharpe, 'sharpe')}. The Sortino ratio is {_fmt_num(sortino)}, "
                "which focuses on downside volatility rather than total volatility."
            ),
            (
                f"The Calmar ratio is {_fmt_num(calmar)}, linking annualized return to maximum "
                "drawdown. A lower Calmar ratio suggests that return generation has come with a "
                "larger path-dependent downside burden."
            ),
            (
                f"The Treynor ratio is {_fmt_num(treynor)} and the Treynor-Black ratio is "
                f"{_fmt_num(treynor_black)}. These values are NaN when the input data does not "
                "include a benchmark_return column."
            ),
        ],
    }

    return commentary


def generate_conclusion_page(analysis: dict) -> List[str]:
    metrics = analysis["metric_categories"]
    risks = metrics["Risk Metrics"]
    adj = metrics["Risk-Adjusted Return Metrics"]

    ann_vol = risks.get("annualized_volatility", np.nan)
    max_dd = risks.get("max_drawdown", np.nan)
    sharpe = adj.get("sharpe_ratio", np.nan)

    conclusion = []

    conclusion.append(
        "Overall, this fund demonstrates a historical risk-return profile that should be interpreted "
        "through both absolute performance and path-dependent downside behavior."
    )

    if pd.notna(sharpe) and pd.notna(max_dd):
        if sharpe >= 1 and max_dd > -0.15:
            conclusion.append(
                "From a historical perspective, the fund appears to have delivered comparatively "
                "efficient returns with manageable drawdown risk."
            )
        elif sharpe >= 0.5 and max_dd > -0.20:
            conclusion.append(
                "From a historical perspective, the fund exhibits a broadly acceptable profile, "
                "though downside episodes and return efficiency remain areas to monitor."
            )
        else:
            conclusion.append(
                "From a historical perspective, the fund's risk-adjusted profile appears less robust, "
                "and further investigation into downside drivers, market sensitivity, and recovery "
                "dynamics would be advisable."
            )

    if pd.notna(ann_vol):
        conclusion.append(
            f"Annualized volatility is {_fmt_pct(ann_vol)}, so ongoing monitoring should focus on "
            "whether future realized volatility remains consistent with the historical pattern."
        )

    conclusion.append(
        "This report is based solely on historical NAV behavior and should be supplemented with "
        "strategy-level due diligence, exposure decomposition, and qualitative review before forming "
        "any investment decision."
    )

    return conclusion


def run_full_analysis(input_path: str) -> dict:
    df = load_nav_data(input_path)

    data_frequency, periods_per_year = infer_data_frequency(df, strict=True)

    df = calculate_daily_returns(df)
    df = calculate_cumulative_returns(df)
    return_metrics = return_summary_metrics(df)
    monthly_return_table, annual_return_table = return_tables(df)

    df, rolling_windows = add_rolling_metrics(df, freq=data_frequency)

    ann_ret = annualized_return(df, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(df, periods_per_year=periods_per_year)
    tail_metrics = tail_risk_metrics(df, periods_per_year=periods_per_year)
    risk_adjusted_metrics = risk_adjusted_return_metrics(df, periods_per_year=periods_per_year)

    df = calculate_drawdown(df)
    drawdown_details = max_drawdown_details(df)
    drawdown_frequencies = drawdown_frequency_summary(df)

    percentage_metrics = _percentage_metric_names(tail_metrics)
    metric_categories = {
        "Data Profile": {
            "input_file": input_path,
            "start_date": df["date"].min(),
            "end_date": df["date"].max(),
            "observations": len(df),
            "data_frequency": data_frequency,
            "annualization_factor": periods_per_year,
            "rolling_windows": ", ".join(str(window) for window in rolling_windows),
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
    risk_signals = generate_risk_signals(metric_categories=metric_categories, df=df)
    metric_categories["Risk Signals"] = risk_signals
    risk_narrative = generate_risk_narrative(risk_signals)
    metric_categories["Risk Narrative"] = {
        "risk_narrative": risk_narrative
    }

    metrics_summary_table = generate_analysis_visualizations(
        df=df,
        metric_categories=metric_categories,
        monthly_return_table=monthly_return_table,
        annual_return_table=annual_return_table,
        drawdown_frequencies=drawdown_frequencies,
        percentage_metrics=percentage_metrics,
    )
    save_monthly_returns_heatmap_table(
        monthly_return_table=monthly_return_table,
        annual_return_table=annual_return_table,
    )

    return {
        "df": df,
        "metric_categories": metric_categories,
        "metrics_summary_table": metrics_summary_table,
        "percentage_metrics": percentage_metrics,
        "monthly_return_table": monthly_return_table,
        "annual_return_table": annual_return_table,
        "drawdown_frequencies": drawdown_frequencies,
        "data_frequency": data_frequency,
        "periods_per_year": periods_per_year,
        "rolling_windows": rolling_windows,
        "risk_signals": risk_signals,
        "risk_narrative": risk_narrative,
    }


def _add_footer(fig, page_number: int) -> None:
    fig.text(0.5, 0.02, f"Page {page_number}", ha="center", va="bottom", fontsize=8, color="#666666")


def _save_cover_page(pdf: PdfPages, analysis: dict, input_path: str, page_number: int) -> None:
    metrics = analysis["metric_categories"]
    risk_metrics = metrics["Risk Metrics"]
    return_metrics = metrics["Return Metrics"]
    risk_adjusted = metrics["Risk-Adjusted Return Metrics"]
    percentage_metrics = analysis["percentage_metrics"]

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.88, "Fund Risk Assessment Report", fontsize=24, weight="bold")
    fig.text(0.08, 0.82, f"Input file: {input_path}", fontsize=11)
    fig.text(0.08, 0.78, f"Generated date: {datetime.now().strftime('%Y-%m-%d')}", fontsize=11)

    profile = metrics["Data Profile"]
    summary_rows = [
        ("Data Frequency", profile["data_frequency"]),
        ("Annualization Factor", _format_metric_value(profile["annualization_factor"], "annualization_factor", percentage_metrics)),
        ("Rolling Windows", profile["rolling_windows"]),
        ("Inception Annualized Return", _format_metric_value(return_metrics["inception_annualized_return"], "inception_annualized_return", percentage_metrics)),
        ("Inception Return", _format_metric_value(return_metrics["inception_return"], "inception_return", percentage_metrics)),
        ("Annualized Volatility", _format_metric_value(risk_metrics["annualized_volatility"], "annualized_volatility", percentage_metrics)),
        ("Maximum Drawdown", _format_metric_value(risk_metrics["max_drawdown"], "max_drawdown", percentage_metrics)),
        ("Maximum Drawdown Date", _format_metric_value(risk_metrics["max_drawdown_date"], "max_drawdown_date", percentage_metrics)),
        ("Sharpe Ratio", _format_metric_value(risk_adjusted["sharpe_ratio"], "sharpe_ratio", percentage_metrics)),
        ("Sortino Ratio", _format_metric_value(risk_adjusted["sortino_ratio"], "sortino_ratio", percentage_metrics)),
        ("Calmar Ratio", _format_metric_value(risk_adjusted["calmar_ratio"], "calmar_ratio", percentage_metrics)),
    ]

    ax = fig.add_axes([0.08, 0.18, 0.84, 0.52])
    ax.axis("off")
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")

    _add_footer(fig, page_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _save_table_pages(
    pdf: PdfPages,
    table_df: pd.DataFrame,
    title: str,
    page_number: int,
    rows_per_page: int = 24,
) -> int:
    if table_df.empty:
        return page_number

    for start in range(0, len(table_df), rows_per_page):
        page_df = table_df.iloc[start : start + rows_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        fig.text(0.06, 0.94, title, fontsize=18, weight="bold")

        table = ax.table(
            cellText=page_df.values,
            colLabels=page_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.18)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#E6E6E6")

        _add_footer(fig, page_number)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_number += 1

    return page_number


def paginate_text(
    paragraphs: List[str],
    characters_per_line: int = 96,
    lines_per_page: int = 30,
) -> List[List[str]]:
    """
    Split paragraphs into page-sized line groups for PDF text pages.
    """
    pages = []
    current_page = []

    for paragraph in paragraphs:
        wrapped_lines = textwrap.wrap(
            paragraph,
            width=characters_per_line,
            break_long_words=False,
            replace_whitespace=False,
        ) or [""]
        wrapped_lines.append("")

        for line in wrapped_lines:
            if len(current_page) >= lines_per_page:
                pages.append(current_page)
                current_page = []

            current_page.append(line)

    if current_page:
        pages.append(current_page)

    return pages


def _save_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: List[str],
    page_number: int,
) -> int:
    """
    Save one logical text section, automatically split across PDF pages if needed.
    """
    pages = paginate_text(paragraphs)

    for index, lines in enumerate(pages):
        page_title = title if index == 0 else f"{title} (continued)"
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.94, page_title, fontsize=18, weight="bold")

        y = 0.86
        for line in lines:
            fig.text(0.08, y, line, fontsize=10.5, va="top", ha="left")
            y -= 0.032 if line else 0.02

        _add_footer(fig, page_number)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_number += 1

    return page_number


def _save_image_page(pdf: PdfPages, image_path: str, title: str, page_number: int) -> int:
    if not os.path.exists(image_path):
        return page_number

    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.06, 0.94, title, fontsize=18, weight="bold")
    _add_footer(fig, page_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return page_number + 1


def _parse_percent_string(value) -> float:
    """
    Convert a formatted percentage string like '12.34%' to a decimal number.
    Returns NaN when the value is blank or invalid.
    """
    if not isinstance(value, str) or value == "" or not value.endswith("%"):
        return np.nan

    try:
        return float(value.rstrip("%")) / 100
    except ValueError:
        return np.nan


def _heatmap_fill_color(value: float, max_abs_value: float) -> str:
    """
    Return a heatmap background color for return values.

    Positive returns use red shades, negative returns use green shades.
    Larger absolute values get deeper colors.
    """
    if np.isnan(value) or max_abs_value <= 0:
        return "#FFFFFF"

    intensity = min(abs(value) / max_abs_value, 1.0)

    # Light base + deeper tone as intensity increases.
    if value > 0:
        red = 255
        green = int(245 - 120 * intensity)
        blue = int(245 - 120 * intensity)
    elif value < 0:
        red = int(245 - 120 * intensity)
        green = 255
        blue = int(245 - 120 * intensity)
    else:
        return "#F7F7F7"

    return f"#{red:02X}{green:02X}{blue:02X}"


def _build_monthly_returns_summary_table(
    monthly_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a year-by-month return table with annual return and monthly win rate.

    Output columns:
    - Year
    - Jan ... Dec
    - Annual Return
    - Win Rate
    """
    if monthly_return_table.empty:
        return pd.DataFrame(
            columns=[
                "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                "Annual Return", "Win Rate",
            ]
        )

    month_data = monthly_return_table.copy()
    month_data["month_period"] = pd.PeriodIndex(month_data["month"], freq="M")
    month_data["Year"] = month_data["month_period"].dt.year
    month_data["month_num"] = month_data["month_period"].dt.month

    month_name_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    month_data["month_name"] = month_data["month_num"].map(month_name_map)

    pivot = month_data.pivot(index="Year", columns="month_name", values="monthly_return")
    ordered_months = list(month_name_map.values())
    pivot = pivot.reindex(columns=ordered_months)

    annual_data = annual_return_table.copy()
    annual_data = annual_data.rename(columns={"year": "Year", "annual_return": "Annual Return"})
    annual_data["Year"] = annual_data["Year"].astype(int)

    win_rate = (
        month_data.groupby("Year")["monthly_return"]
        .apply(lambda values: (values > 0).mean() if len(values) else np.nan)
        .rename("Win Rate")
    )

    summary = pivot.join(annual_data.set_index("Year"), how="left").join(win_rate, how="left")
    summary = summary.reset_index()

    for col in ordered_months + ["Annual Return", "Win Rate"]:
        if col in summary.columns:
            summary[col] = summary[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )

    return summary


def _save_return_charts_and_table_section(
    pdf: PdfPages,
    annual_image_path: str,
    monthly_image_path: str,
    monthly_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    page_number: int,
) -> int:
    """
    Save one combined return-analysis page:
    - upper half: annual return chart and monthly return chart
    - lower half: year-by-month return heatmap table
    """
    summary_table = _build_monthly_returns_summary_table(
        monthly_return_table=monthly_return_table,
        annual_return_table=annual_return_table,
    )

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, "Return Analysis", fontsize=18, weight="bold")

    if os.path.exists(annual_image_path):
        annual_image = plt.imread(annual_image_path)
        ax_annual_chart = fig.add_axes([0.04, 0.50, 0.44, 0.36])
        ax_annual_chart.imshow(annual_image)
        ax_annual_chart.axis("off")
        fig.text(0.04, 0.88, "Annual Return", fontsize=11, weight="bold")

    if os.path.exists(monthly_image_path):
        monthly_image = plt.imread(monthly_image_path)
        ax_monthly_chart = fig.add_axes([0.52, 0.50, 0.44, 0.36])
        ax_monthly_chart.imshow(monthly_image)
        ax_monthly_chart.axis("off")
        fig.text(0.52, 0.88, "Monthly Return", fontsize=11, weight="bold")

    ax_table = fig.add_axes([0.05, 0.05, 0.90, 0.36])
    ax_table.axis("off")

    if summary_table.empty:
        ax_table.text(0.0, 0.9, "No monthly return table is available.", fontsize=10.5)
    else:
        table = ax_table.table(
            cellText=summary_table.values,
            colLabels=summary_table.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.2)

        month_columns = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        annual_return_col = summary_table.columns.get_loc("Annual Return")
        win_rate_col = summary_table.columns.get_loc("Win Rate")
        return_columns = month_columns | {"Annual Return"}
        heatmap_values = []

        for _, row_data in summary_table.iterrows():
            for column_name in summary_table.columns:
                if column_name in return_columns:
                    parsed_value = _parse_percent_string(row_data[column_name])
                    if not np.isnan(parsed_value):
                        heatmap_values.append(parsed_value)

        max_abs_value = max((abs(value) for value in heatmap_values), default=0.0)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#E6E6E6")
                if col == annual_return_col:
                    cell.set_facecolor("#E6E6E6")
                continue

            cell_value = summary_table.iloc[row - 1, col]
            column_name = summary_table.columns[col]

            if col == annual_return_col:
                cell.set_text_props(weight="bold")

            if col == win_rate_col:
                cell.set_text_props(weight="bold")

            if column_name in return_columns:
                numeric_value = _parse_percent_string(cell_value)
                cell.set_facecolor(_heatmap_fill_color(numeric_value, max_abs_value))
                if not np.isnan(numeric_value):
                    cell.get_text().set_color("#202020")

            if column_name == "Year":
                cell.set_facecolor("#F5F5F5")
                cell.set_text_props(weight="bold")

    _add_footer(fig, page_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return page_number + 1


def save_monthly_returns_heatmap_table(
    monthly_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/monthly_returns_heatmap_table.png",
) -> None:
    """
    Save the yearly monthly-return heatmap table as a standalone chart image.
    """
    os.makedirs(CHART_DIR, exist_ok=True)
    summary_table = _build_monthly_returns_summary_table(
        monthly_return_table=monthly_return_table,
        annual_return_table=annual_return_table,
    )

    fig, ax = plt.subplots(figsize=(16, 4.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.text(0.05, 0.93, "Monthly Returns Heatmap Table", fontsize=16, weight="bold")

    if summary_table.empty:
        ax.text(0.0, 0.85, "No monthly return table is available.", fontsize=10.5)
    else:
        table = ax.table(
            cellText=summary_table.values,
            colLabels=summary_table.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1, 1.4)

        month_columns = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        annual_return_col = summary_table.columns.get_loc("Annual Return")
        win_rate_col = summary_table.columns.get_loc("Win Rate")
        return_columns = month_columns | {"Annual Return"}
        heatmap_values = []

        for _, row_data in summary_table.iterrows():
            for column_name in summary_table.columns:
                if column_name in return_columns:
                    parsed_value = _parse_percent_string(row_data[column_name])
                    if not np.isnan(parsed_value):
                        heatmap_values.append(parsed_value)

        max_abs_value = max((abs(value) for value in heatmap_values), default=0.0)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#E6E6E6")
                continue

            cell_value = summary_table.iloc[row - 1, col]
            column_name = summary_table.columns[col]

            if col == annual_return_col or col == win_rate_col or column_name == "Year":
                cell.set_text_props(weight="bold")

            if column_name in return_columns:
                numeric_value = _parse_percent_string(cell_value)
                cell.set_facecolor(_heatmap_fill_color(numeric_value, max_abs_value))
                if not np.isnan(numeric_value):
                    cell.get_text().set_color("#202020")

            if column_name == "Year":
                cell.set_facecolor("#F5F5F5")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(analysis: dict, input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metric_table = _flatten_metric_categories(
        analysis["metric_categories"],
        percentage_metrics=analysis["percentage_metrics"],
    )

    chart_pages: List[Tuple[str, str]] = [
        (f"{CHART_DIR}/nav_drawdown.png", "NAV and Drawdown"),
        (f"{CHART_DIR}/rolling_metrics.png", "Rolling Risk Metrics"),
        (f"{CHART_DIR}/drawdown_frequency.png", "Drawdown Frequency"),
    ]

    with PdfPages(output_path) as pdf:
        page_number = 1
        _save_cover_page(pdf, analysis, input_path, page_number)
        page_number += 1

        commentary = generate_report_commentary(analysis)
        for section_title, paragraphs in commentary.items():
            page_number = _save_text_page(pdf, section_title, paragraphs, page_number)

        page_number = _save_table_pages(pdf, metric_table, "Metrics Summary", page_number)
        page_number = _save_text_page(
            pdf,
            "Risk Commentary",
            [analysis["risk_narrative"]],
            page_number,
        )

        page_number = _save_return_charts_and_table_section(
            pdf,
            annual_image_path=f"{CHART_DIR}/annual_returns.png",
            monthly_image_path=f"{CHART_DIR}/monthly_returns.png",
            monthly_return_table=analysis["monthly_return_table"],
            annual_return_table=analysis["annual_return_table"],
            page_number=page_number,
        )

        for chart_path, title in chart_pages:
            page_number = _save_image_page(pdf, chart_path, title, page_number)

        conclusion = generate_conclusion_page(analysis)
        page_number = _save_text_page(pdf, "Conclusion and Risk Notes", conclusion, page_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF fund risk assessment report.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Input NAV file path. Supports CSV, XLSX, and XLS. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT_PATH}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = run_full_analysis(args.input)
    build_pdf_report(analysis, input_path=args.input, output_path=args.output)

    print(f"PDF report generated: {args.output}")
    print(f"Metrics CSV generated: {REPORT_DIR}/metrics_summary_table.csv")
    print(f"Charts generated under: {CHART_DIR}")


if __name__ == "__main__":
    main()
