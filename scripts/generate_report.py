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

from src.analysis_pipeline import run_analysis_pipeline
from src.visualization import (
    CHART_DIR,
    REPORT_DIR,
    _resolve_output_dirs,
    build_monthly_returns_heatmap_table,
    draw_monthly_returns_heatmap_table,
    save_metrics_summary_table_image,
    split_metric_categories_for_pages,
)


DEFAULT_INPUT_PATH = "data/sample_nav_data.xlsx"
DEFAULT_OUTPUT_PATH = f"{REPORT_DIR}/fund_risk_report.pdf"
DEFAULT_BENCHMARK_CANDIDATES = (
    "data/benchmark.xlsx",
    "data/benchmark.csv",
    "data/benchmark.xls",
)


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


def _display_metric_label(metric_name: str) -> str:
    label = str(metric_name).replace("_", " ").title()
    return label.replace("Cvar", "CVaR").replace("Var", "VaR").replace("Es", "ES")


def _flatten_metric_categories(
    metric_categories: Dict[str, Dict[str, object]],
    percentage_metrics: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for category, metrics in metric_categories.items():
        for index, (metric_name, metric_value) in enumerate(metrics.items()):
            rows.append(
                {
                    "Category": category if index == 0 else "",
                    "Metric": _display_metric_label(metric_name),
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
    benchmark = metrics.get("Benchmark Comparison", {})
    tail = metrics["Tail Risk Metrics"]
    adj = metrics["Risk-Adjusted Return"]
    benchmark_name = analysis.get("benchmark_name", profile.get("benchmark_name", "Benchmark"))

    data_frequency = profile.get("frequency", "unknown")
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
    benchmark_return = benchmark.get("benchmark_annualized_return", np.nan)
    annualized_excess_return = benchmark.get("annualized_excess_return", np.nan)
    cumulative_excess_return = benchmark.get("cumulative_excess_return", np.nan)
    tracking_error = benchmark.get("tracking_error", np.nan)
    information_ratio = benchmark.get("information_ratio", np.nan)
    df = analysis.get("df", pd.DataFrame())
    clean_returns = (
        pd.to_numeric(df.get("fund_return"), errors="coerce").dropna()
        if "fund_return" in df.columns
        else pd.Series(dtype=float)
    )

    var_95 = tail.get("var_95_period", np.nan)
    var_99 = tail.get("var_99_period", np.nan)
    cvar_95 = tail.get("cvar_es_95_period", np.nan)
    cvar_99 = tail.get("cvar_es_99_period", np.nan)

    q95_return = clean_returns.quantile(0.05) if len(clean_returns) else np.nan
    q99_return = clean_returns.quantile(0.01) if len(clean_returns) else np.nan
    q95_tail_count = int((clean_returns <= q95_return).sum()) if len(clean_returns) else 0
    q99_tail_count = int((clean_returns <= q99_return).sum()) if len(clean_returns) else 0

    var_levels_overlap = (
        pd.notna(var_95)
        and pd.notna(var_99)
        and np.isclose(var_95, var_99, rtol=1e-6, atol=1e-6)
    )
    cvar_levels_overlap = (
        pd.notna(cvar_95)
        and pd.notna(cvar_99)
        and np.isclose(cvar_95, cvar_99, rtol=1e-6, atol=1e-6)
    )

    if pd.isna(annualized_excess_return):
        benchmark_summary_sentence = (
            f"{benchmark_name} delivered an annualized return of {_fmt_pct(benchmark_return)} over the aligned sample, "
            "while annualized excess return is not available."
        )
    else:
        benchmark_direction = "outperformed" if annualized_excess_return >= 0 else "underperformed"
        benchmark_gap = _fmt_pct(abs(annualized_excess_return))
        benchmark_summary_sentence = (
            f"{benchmark_name} delivered an annualized return of {_fmt_pct(benchmark_return)} over the aligned sample, "
            f"and the fund {benchmark_direction} {benchmark_name} by {benchmark_gap} on an annualized basis."
        )

    if var_levels_overlap:
        var_interpretation_sentence = (
            f"The 95% one-period VaR is {_fmt_pct(var_95)}, and the corresponding 99% one-period VaR is also {_fmt_pct(var_99)}. "
            "In this sample, both confidence levels map to the same extreme historical observation, which can occur when the return history is limited or observed at a lower frequency such as weekly data. "
            "This indicates that tail risk differentiation is limited within the available sample, so the incremental severity normally associated with the 99% threshold is not observable here."
        )
    else:
        var_interpretation_sentence = (
            f"The 95% one-period VaR is {_fmt_pct(var_95)}, while the 99% one-period VaR is {_fmt_pct(var_99)}. "
            "The 99% confidence level isolates a narrower and typically more adverse segment of the historical loss distribution."
        )

    if cvar_levels_overlap:
        cvar_interpretation_sentence = (
            f"The 95% one-period CVaR/ES is {_fmt_pct(cvar_95)}, while the 99% one-period CVaR/ES is {_fmt_pct(cvar_99)}. "
            "This indicates that the tail contains very few distinct observations; in sparse samples, CVaR can converge to VaR when only a small number of returns fall into the tail. "
            "In that setting, VaR-based measures alone provide limited differentiation across extreme loss states and may understate the severity of downside outcomes beyond the observed tail."
        )
    else:
        cvar_interpretation_sentence = (
            f"The 95% one-period CVaR/ES is {_fmt_pct(cvar_95)}, while the 99% one-period CVaR/ES is {_fmt_pct(cvar_99)}. "
            "CVaR/ES measures the average realized loss within the tail and therefore complements VaR by describing loss severity beyond the threshold."
        )

    tail_warning_sentence = (
        f"Tail risk estimates are based on a limited number of tail observations in this sample "
        f"(approximately {q95_tail_count} observations at the 95% threshold and {q99_tail_count} at the 99% threshold) "
        "and may not fully capture extreme loss scenarios. In such cases, greater weight should be placed on drawdown analysis, volatility persistence, and other path-dependent measures when assessing downside risk."
        if q95_tail_count < 5 or q99_tail_count < 5
        else (
            "Tail risk estimates should be interpreted with caution because their stability depends on both sample depth and data frequency, and complementary evidence from drawdown behaviour and volatility persistence remains important for risk assessment."
        )
    )

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
                "The fund NAV date series defines the analysis timeline. Benchmark levels are "
                f"aligned to those dates using the most recent {benchmark_name} observation on or before each fund date."
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
        "Benchmark Relative Commentary": [
            (benchmark_summary_sentence),
            (
                "Monthly Excess Return measures period-by-period outperformance versus the benchmark, "
                "while Cumulative Excess Return is defined as cumulative fund return minus cumulative benchmark return."
            ),
            (
                f"Cumulative excess return over the aligned sample is {_fmt_pct(cumulative_excess_return)}. "
                f"Tracking error is {_fmt_pct(tracking_error)}, and the information ratio is {_fmt_num(information_ratio)}. "
                f"These measures summarize both the magnitude and efficiency of active return relative to {benchmark_name}."
            ),
            (
                f"Benchmark data is aligned to fund NAV dates using the most recent available {benchmark_name} observation "
                "on or before each fund date."
            ),
        ],
        "Tail Risk Commentary": [
            (
                f"{var_interpretation_sentence} "
                f"The corresponding 10-period VaR at 95% is {_fmt_pct(tail.get('var_95_10_period', np.nan))}."
            ),
            (
                f"The corresponding 10-period VaR at 99% is {_fmt_pct(tail.get('var_99_10_period', np.nan))}. "
                f"These figures are estimated under historical simulation using the native {data_frequency} return frequency."
            ),
            (
                f"{cvar_interpretation_sentence} {tail_warning_sentence}"
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
                f"{_fmt_num(treynor_black)}. These metrics depend on the aligned benchmark return series."
            ),
        ],
    }

    return commentary


def generate_conclusion_page(analysis: dict) -> List[str]:
    metrics = analysis["metric_categories"]
    risks = metrics["Risk Metrics"]
    adj = metrics["Risk-Adjusted Return"]

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


def run_full_analysis(
    input_path: str,
    benchmark_path: str,
    output_root: str = "output",
    benchmark_name: str | None = None,
) -> dict:
    """
    Run the shared benchmark-aware analysis pipeline for report generation.

    The fund NAV dates define the target timeline, and benchmark levels are
    aligned backward onto those same dates before returns are calculated.
    """
    analysis = run_analysis_pipeline(
        fund_input_path=input_path,
        benchmark_input_path=benchmark_path,
        output_dir=output_root,
        benchmark_name=benchmark_name,
    )
    save_monthly_returns_heatmap_table(
        monthly_return_table=analysis["monthly_return_table"],
        benchmark_monthly_return_table=analysis["benchmark_monthly_return_table"],
        monthly_excess_return_table=analysis["monthly_excess_return_table"],
        annual_return_table=analysis["annual_return_table"],
        benchmark_annual_return_table=analysis["benchmark_annual_return_table"],
        annual_excess_return_table=analysis["annual_excess_return_table"],
        output_path=os.path.join(_resolve_output_dirs(output_root)[0], "monthly_returns_heatmap_table.png"),
    )
    return analysis


def _add_footer(fig, page_number: int) -> None:
    fig.text(0.5, 0.02, f"Page {page_number}", ha="center", va="bottom", fontsize=8, color="#666666")


def _save_cover_page(pdf: PdfPages, analysis: dict, input_path: str, page_number: int) -> None:
    metrics = analysis["metric_categories"]
    risk_metrics = metrics["Risk Metrics"]
    return_metrics = metrics["Return Metrics"]
    benchmark_metrics = metrics.get("Benchmark Comparison", {})
    risk_adjusted = metrics["Risk-Adjusted Return"]
    percentage_metrics = analysis["percentage_metrics"]
    benchmark_name = analysis.get("benchmark_name", metrics["Data Profile"].get("benchmark_name", "Benchmark"))

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.88, f"Fund Risk Assessment Report vs {benchmark_name}", fontsize=24, weight="bold")

    profile = metrics["Data Profile"]
    summary_rows = [
        ("Data Frequency", profile["frequency"]),
        ("Annualization Factor", _format_metric_value(profile["annualization_factor"], "annualization_factor", percentage_metrics)),
        ("Rolling Windows", profile["rolling_windows"]),
        ("Inception Annualized Return", _format_metric_value(return_metrics["inception_annualized_return"], "inception_annualized_return", percentage_metrics)),
        ("Inception Return", _format_metric_value(return_metrics["inception_return"], "inception_return", percentage_metrics)),
        (f"{benchmark_name} Annualized Return", _format_metric_value(benchmark_metrics.get("benchmark_annualized_return", np.nan), "benchmark_annualized_return", percentage_metrics)),
        ("Annualized Excess Return", _format_metric_value(benchmark_metrics.get("annualized_excess_return", np.nan), "annualized_excess_return", percentage_metrics)),
        ("Cumulative Excess Return", _format_metric_value(benchmark_metrics.get("cumulative_excess_return", np.nan), "cumulative_excess_return", percentage_metrics)),
        ("Annualized Volatility", _format_metric_value(risk_metrics["annualized_volatility"], "annualized_volatility", percentage_metrics)),
        ("Maximum Drawdown", _format_metric_value(risk_metrics["max_drawdown"], "max_drawdown", percentage_metrics)),
        ("Maximum Drawdown Date", _format_metric_value(risk_metrics["max_drawdown_date"], "max_drawdown_date", percentage_metrics)),
        ("Tracking Error", _format_metric_value(benchmark_metrics.get("tracking_error", np.nan), "tracking_error", percentage_metrics)),
        ("Information Ratio", _format_metric_value(benchmark_metrics.get("information_ratio", np.nan), "information_ratio", percentage_metrics)),
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


def _save_return_charts_and_table_section(
    pdf: PdfPages,
    annual_image_path: str,
    monthly_image_path: str,
    monthly_return_table: pd.DataFrame,
    benchmark_monthly_return_table: pd.DataFrame,
    monthly_excess_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_annual_return_table: pd.DataFrame,
    annual_excess_return_table: pd.DataFrame,
    page_number: int,
) -> int:
    """
    Save one combined return-analysis page:
    - upper half: annual return chart and monthly return chart
    - lower half: year-by-month return heatmap table
    """
    summary_table = build_monthly_returns_heatmap_table(
        monthly_return_table=monthly_return_table,
        benchmark_monthly_return_table=benchmark_monthly_return_table,
        monthly_excess_return_table=monthly_excess_return_table,
        annual_return_table=annual_return_table,
        benchmark_annual_return_table=benchmark_annual_return_table,
        annual_excess_return_table=annual_excess_return_table,
    )

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, "Return Analysis", fontsize=18, weight="bold")
    fig.text(
        0.06,
        0.92,
        "Monthly excess return = monthly fund return minus monthly benchmark return. "
        "Cumulative excess return is shown separately on the NAV chart as cumulative fund minus cumulative benchmark.",
        fontsize=9,
    )

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

    ax_table = fig.add_axes([0.05, 0.04, 0.90, 0.34])
    ax_table.axis("off")

    if summary_table.empty:
        ax_table.text(0.0, 0.9, "No monthly return table is available.", fontsize=10.5)
    else:
        draw_monthly_returns_heatmap_table(
            ax=ax_table,
            summary_table=summary_table,
            bbox=(0.0, 0.0, 1.0, 0.88),
            header_fontsize=7.5,
            body_fontsize=7.1,
        )

    _add_footer(fig, page_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return page_number + 1


def _save_large_image_page(
    pdf: PdfPages,
    image_path: str,
    title: str,
    page_number: int,
    figsize: Tuple[float, float] = (13.5, 8.5),
) -> int:
    """
    Save one dedicated image page with the image scaled to occupy most of the
    available page area while preserving aspect ratio.
    """
    if not os.path.exists(image_path):
        return page_number

    image = plt.imread(image_path)
    image_height, image_width = image.shape[:2]
    image_aspect = image_width / image_height if image_height else 1.0

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.95, title, fontsize=18, weight="bold", ha="left", va="top")

    left_margin = 0.035
    right_margin = 0.035
    bottom_margin = 0.055
    top_reserved = 0.11
    available_width = 1.0 - left_margin - right_margin
    available_height = 1.0 - top_reserved - bottom_margin
    available_aspect = (figsize[0] * available_width) / (figsize[1] * available_height)

    if image_aspect >= available_aspect:
        axes_width = available_width
        axes_height = axes_width / image_aspect * (figsize[0] / figsize[1])
    else:
        axes_height = available_height
        axes_width = axes_height * image_aspect * (figsize[1] / figsize[0])

    axes_left = left_margin + (available_width - axes_width) / 2
    axes_bottom = bottom_margin + (available_height - axes_height) / 2

    ax = fig.add_axes([axes_left, axes_bottom, axes_width, axes_height])
    ax.imshow(image, aspect="auto")
    ax.axis("off")

    _add_footer(fig, page_number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return page_number + 1


def save_monthly_returns_heatmap_table(
    monthly_return_table: pd.DataFrame,
    benchmark_monthly_return_table: pd.DataFrame,
    monthly_excess_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_annual_return_table: pd.DataFrame,
    annual_excess_return_table: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/monthly_returns_heatmap_table.png",
) -> None:
    """
    Save the yearly fund/benchmark/excess monthly-return heatmap table as a standalone chart image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_table = build_monthly_returns_heatmap_table(
        monthly_return_table=monthly_return_table,
        benchmark_monthly_return_table=benchmark_monthly_return_table,
        monthly_excess_return_table=monthly_excess_return_table,
        annual_return_table=annual_return_table,
        benchmark_annual_return_table=benchmark_annual_return_table,
        annual_excess_return_table=annual_excess_return_table,
    )

    fig, ax = plt.subplots(figsize=(17, 5.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.suptitle(
        "Monthly Fund, Benchmark, and Monthly Excess Return Heatmap Table",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    if summary_table.empty:
        ax.text(0.0, 0.85, "No monthly return table is available.", fontsize=10.5)
    else:
        draw_monthly_returns_heatmap_table(
            ax=ax,
            summary_table=summary_table,
            bbox=(0.0, 0.0, 1.0, 0.84),
            header_fontsize=8.2,
            body_fontsize=7.8,
        )

    fig.subplots_adjust(top=0.86, bottom=0.05)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(analysis: dict, input_path: str, output_path: str, output_root: str = "output") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chart_dir, report_dir = _resolve_output_dirs(output_root)

    chart_pages: List[Tuple[str, str]] = [
        (os.path.join(chart_dir, "nav_drawdown.png"), "Fund, Benchmark, Cumulative Excess Return, and Drawdown"),
        (os.path.join(chart_dir, "rolling_metrics.png"), "Rolling Risk Metrics"),
        (os.path.join(chart_dir, "drawdown_frequency.png"), "Drawdown Frequency"),
    ]

    with PdfPages(output_path) as pdf:
        page_number = 1
        _save_cover_page(pdf, analysis, input_path, page_number)
        page_number += 1

        commentary = generate_report_commentary(analysis)
        for section_title, paragraphs in commentary.items():
            page_number = _save_text_page(pdf, section_title, paragraphs, page_number)

        metrics_summary_pages = split_metric_categories_for_pages(
            analysis["metric_categories"],
            percentage_metrics=analysis["percentage_metrics"],
            max_total_units_per_page=19.5,
        )
        for page_index, metric_categories_page in enumerate(metrics_summary_pages, start=1):
            page_image_path = os.path.join(
                chart_dir,
                f"metrics_summary_table_page_{page_index}.png",
            )
            save_metrics_summary_table_image(
                metric_categories=metric_categories_page,
                output_path=page_image_path,
                percentage_metrics=analysis["percentage_metrics"],
            )
            title = "Metrics Summary" if len(metrics_summary_pages) == 1 else f"Metrics Summary ({page_index}/{len(metrics_summary_pages)})"
            page_number = _save_large_image_page(
                pdf,
                page_image_path,
                title,
                page_number,
            )
        page_number = _save_text_page(
            pdf,
            "Risk Commentary",
            [analysis["risk_narrative"]],
            page_number,
        )

        page_number = _save_return_charts_and_table_section(
            pdf,
            annual_image_path=os.path.join(chart_dir, "annual_returns.png"),
            monthly_image_path=os.path.join(chart_dir, "monthly_returns.png"),
            monthly_return_table=analysis["monthly_return_table"],
            benchmark_monthly_return_table=analysis["benchmark_monthly_return_table"],
            monthly_excess_return_table=analysis["monthly_excess_return_table"],
            annual_return_table=analysis["annual_return_table"],
            benchmark_annual_return_table=analysis["benchmark_annual_return_table"],
            annual_excess_return_table=analysis["annual_excess_return_table"],
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
        "--benchmark",
        default=None,
        help="Benchmark level file path. The second column must be a benchmark level series, not returns.",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Optional friendly benchmark name to display in charts, tables, commentary, and the PDF report.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output root for charts and CSV artifacts. Default: output",
    )

    return parser.parse_args()


def resolve_benchmark_path(benchmark_arg: str | None) -> str:
    """
    Resolve the benchmark file path from CLI input or standard data locations.
    """
    if benchmark_arg:
        return benchmark_arg

    for candidate in DEFAULT_BENCHMARK_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    raise ValueError(
        "Benchmark input is required for report generation. "
        "Pass --benchmark or add data/benchmark.xlsx, data/benchmark.csv, or data/benchmark.xls."
    )


def main() -> None:
    args = parse_args()
    benchmark_path = resolve_benchmark_path(args.benchmark)
    analysis = run_full_analysis(
        args.input,
        benchmark_path=benchmark_path,
        output_root=args.output_dir,
        benchmark_name=args.benchmark_name,
    )
    build_pdf_report(
        analysis,
        input_path=args.input,
        output_path=args.output,
        output_root=args.output_dir,
    )

    print(f"PDF report generated: {args.output}")
    print(f"Benchmark file used: {benchmark_path}")
    print(f"Benchmark name used: {analysis['benchmark_name']}")
    print(f"Metrics CSV generated: {os.path.join(_resolve_output_dirs(args.output_dir)[1], 'metrics_summary_table.csv')}")
    print(f"Charts generated under: {_resolve_output_dirs(args.output_dir)[0]}")


if __name__ == "__main__":
    main()
