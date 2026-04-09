import os
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

CHART_DIR = "output/charts"
REPORT_DIR = "output/reports"


def _resolve_output_dirs(output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve chart and report directories, optionally under a custom output root.
    """
    if output_dir:
        chart_dir = os.path.join(output_dir, "charts")
        report_dir = os.path.join(output_dir, "reports")
        return chart_dir, report_dir

    return CHART_DIR, REPORT_DIR


def _ensure_output_dirs(output_dir: Optional[str] = None) -> Tuple[str, str]:
    chart_dir, report_dir = _resolve_output_dirs(output_dir=output_dir)
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    return chart_dir, report_dir


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

    if "months" in metric_name or "days" in metric_name or metric_name == "observations":
        return f"{value:.0f}"

    if metric_name in percentage_metrics:
        return f"{value:.2%}"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}"

    return str(value)


def _display_metric_label(metric_name: str) -> str:
    """
    Convert internal metric identifiers into user-facing labels.
    """
    label = str(metric_name).replace("_", " ").title()
    return label.replace("Cvar", "CVaR").replace("Var", "VaR").replace("Es", "ES")


def _normalize_metric_categories(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
) -> List[Tuple[str, List[Tuple[str, object]]]]:
    """
    Normalize summary metrics into grouped category rows.
    """
    grouped_rows: List[Tuple[str, List[Tuple[str, object]]]] = []
    for category, metrics in metric_categories.items():
        if isinstance(metrics, dict):
            rows = list(metrics.items())
        else:
            rows = list(metrics)

        grouped_rows.append((category, rows))

    return grouped_rows


def split_metric_categories_for_pages(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    percentage_metrics: Optional[Iterable[str]] = None,
    max_total_units_per_page: float = 22.0,
) -> List[Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]]]:
    """
    Split metric categories into page-sized contiguous groups for chart-style
    PDF rendering while preserving merged category blocks whenever practical.
    If a single category is too tall, it is split into page-sized segments.
    """
    percentage_metrics = set(percentage_metrics or [])
    grouped_rows = _normalize_metric_categories(metric_categories)
    if not grouped_rows:
        return [{}]

    def estimate_row_units(metric_name: str, metric_value) -> float:
        metric_text = _display_metric_label(metric_name)
        value_text = _format_metric_value(metric_value, metric_name, percentage_metrics)
        metric_lines = _wrapped_lines(metric_text, width=28)
        value_lines = _wrapped_lines(value_text, width=34)
        line_count = max(len(metric_lines), len(value_lines))
        return max(1.0, 0.9 + 0.45 * (line_count - 1))

    pages: List[Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]]] = []
    current_page: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]] = {}
    current_units = 1.0  # header row

    def flush_page() -> None:
        nonlocal current_page, current_units
        if current_page:
            pages.append(current_page)
            current_page = {}
            current_units = 1.0

    for category, rows in grouped_rows:
        row_units = [estimate_row_units(metric_name, metric_value) for metric_name, metric_value in rows]
        category_units = sum(row_units)

        if current_page and current_units + category_units > max_total_units_per_page:
            flush_page()

        if category_units <= max_total_units_per_page - 1.0:
            current_page[category] = dict(rows)
            current_units += category_units
            continue

        partial_rows: List[Tuple[str, object]] = []
        partial_units = 0.0
        for (metric_name, metric_value), unit in zip(rows, row_units):
            if partial_rows and current_units + partial_units + unit > max_total_units_per_page:
                current_page[category] = dict(partial_rows)
                flush_page()
                partial_rows = []
                partial_units = 0.0

            partial_rows.append((metric_name, metric_value))
            partial_units += unit

        if partial_rows:
            current_page[category] = dict(partial_rows)
            current_units += partial_units

    flush_page()
    return pages or [{}]


def build_metrics_summary_table(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    percentage_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build a grouped metrics summary table for CSV export and reporting.
    """
    percentage_metrics = set(percentage_metrics or [])
    rows = []

    for category, metrics in _normalize_metric_categories(metric_categories):
        for index, (metric_name, metric_value) in enumerate(metrics):
            rows.append(
                {
                    "category": category if index == 0 else "",
                    "metric": _display_metric_label(metric_name),
                    "value": _format_metric_value(
                        metric_value,
                        metric_name,
                        percentage_metrics,
                    ),
                }
            )

    return pd.DataFrame(rows, columns=["category", "metric", "value"])


def _wrapped_lines(text: str, width: int) -> List[str]:
    wrapped = textwrap.wrap(
        str(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [""]


def _parse_percent_string(value) -> float:
    """
    Convert a formatted percentage string like '12.34%' to decimal form.
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
    """
    if np.isnan(value) or max_abs_value <= 0:
        return "#FFFFFF"

    intensity = min(abs(value) / max_abs_value, 1.0)
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


def build_monthly_returns_heatmap_table(
    monthly_return_table: pd.DataFrame,
    benchmark_monthly_return_table: pd.DataFrame,
    monthly_excess_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_annual_return_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a three-row-per-year heatmap table with fund, benchmark, and excess monthly returns.
    """
    columns = [
        "Year", "Row Type", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Annual Return", "Win Rate",
    ]
    if monthly_return_table.empty:
        return pd.DataFrame(columns=columns)

    month_name_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    ordered_months = list(month_name_map.values())

    def _pivot_monthly(table: pd.DataFrame, value_col: str) -> pd.DataFrame:
        data = table.copy()
        data["month_period"] = pd.PeriodIndex(data["month"], freq="M")
        data["Year"] = data["month_period"].dt.year
        data["month_name"] = data["month_period"].dt.month.map(month_name_map)
        return data.pivot(index="Year", columns="month_name", values=value_col).reindex(columns=ordered_months)

    fund_pivot = _pivot_monthly(monthly_return_table, "monthly_return")
    benchmark_pivot = _pivot_monthly(benchmark_monthly_return_table, "benchmark_monthly_return")
    excess_pivot = _pivot_monthly(monthly_excess_return_table, "monthly_excess_return")

    annual_data = annual_return_table.copy().rename(
        columns={"year": "Year", "annual_return": "Annual Return"}
    )
    annual_data["Year"] = annual_data["Year"].astype(int)
    annual_data = annual_data.set_index("Year")
    benchmark_annual_data = benchmark_annual_return_table.copy().rename(
        columns={"year": "Year", "benchmark_annual_return": "Annual Return"}
    )
    benchmark_annual_data["Year"] = benchmark_annual_data["Year"].astype(int)
    benchmark_annual_data = benchmark_annual_data.set_index("Year")

    fund_win_rate = (
        monthly_return_table.assign(month_period=pd.PeriodIndex(monthly_return_table["month"], freq="M"))
        .assign(Year=lambda x: x["month_period"].dt.year)
        .groupby("Year")["monthly_return"]
        .apply(lambda values: (values > 0).mean() if len(values) else np.nan)
        .rename("Win Rate")
    )
    benchmark_win_rate = (
        benchmark_monthly_return_table.assign(month_period=pd.PeriodIndex(benchmark_monthly_return_table["month"], freq="M"))
        .assign(Year=lambda x: x["month_period"].dt.year)
        .groupby("Year")["benchmark_monthly_return"]
        .apply(lambda values: (values > 0).mean() if len(values) else np.nan)
        .rename("Win Rate")
    )
    excess_annual = (
        annual_data["Annual Return"].rename("fund_annual_return").to_frame()
        .join(benchmark_annual_data["Annual Return"].rename("benchmark_annual_return"), how="outer")
        .assign(**{"Annual Return": lambda x: x["fund_annual_return"] - x["benchmark_annual_return"]})["Annual Return"]
        .rename("Annual Return")
    )
    excess_win_rate = (
        monthly_excess_return_table.assign(month_period=pd.PeriodIndex(monthly_excess_return_table["month"], freq="M"))
        .assign(Year=lambda x: x["month_period"].dt.year)
        .groupby("Year")
        .apply(
            lambda group: (
                (group.dropna(subset=["monthly_return", "benchmark_monthly_return"])["monthly_return"]
                 > group.dropna(subset=["monthly_return", "benchmark_monthly_return"])["benchmark_monthly_return"]).mean()
                if len(group.dropna(subset=["monthly_return", "benchmark_monthly_return"]))
                else np.nan
            )
        )
        .rename("Win Rate")
    )

    all_years = sorted(set(fund_pivot.index) | set(benchmark_pivot.index) | set(excess_pivot.index))
    rows = []
    for year in all_years:
        fund_row = {"Year": str(year), "Row Type": "Fund Return"}
        for month_name in ordered_months:
            fund_row[month_name] = fund_pivot.loc[year, month_name] if year in fund_pivot.index else np.nan
        fund_row["Annual Return"] = annual_data.loc[year, "Annual Return"] if year in annual_data.index else np.nan
        fund_row["Win Rate"] = fund_win_rate.loc[year] if year in fund_win_rate.index else np.nan
        rows.append(fund_row)

        benchmark_row = {"Year": "", "Row Type": "Benchmark Return"}
        for month_name in ordered_months:
            benchmark_row[month_name] = benchmark_pivot.loc[year, month_name] if year in benchmark_pivot.index else np.nan
        benchmark_row["Annual Return"] = (
            benchmark_annual_data.loc[year, "Annual Return"] if year in benchmark_annual_data.index else np.nan
        )
        benchmark_row["Win Rate"] = benchmark_win_rate.loc[year] if year in benchmark_win_rate.index else np.nan
        rows.append(benchmark_row)

        excess_row = {"Year": "", "Row Type": "Excess Return"}
        for month_name in ordered_months:
            excess_row[month_name] = excess_pivot.loc[year, month_name] if year in excess_pivot.index else np.nan
        excess_row["Annual Return"] = excess_annual.loc[year] if year in excess_annual.index else np.nan
        excess_row["Win Rate"] = excess_win_rate.loc[year] if year in excess_win_rate.index else np.nan
        rows.append(excess_row)

    summary = pd.DataFrame(rows, columns=columns)
    for col in ordered_months + ["Annual Return", "Win Rate"]:
        summary[col] = summary[col].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")

    return summary


def draw_monthly_returns_heatmap_table(
    ax: plt.Axes,
    summary_table: pd.DataFrame,
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    header_fontsize: float = 8.5,
    body_fontsize: float = 8.0,
) -> None:
    """
    Draw the monthly fund / benchmark / excess return heatmap table with a
    vertically merged Year column.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if summary_table.empty:
        ax.text(bbox[0], bbox[1] + bbox[3] * 0.9, "No monthly return table is available.", fontsize=10.5)
        return

    left, bottom, width, height = bbox
    columns = list(summary_table.columns)
    month_columns = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    return_columns = month_columns | {"Annual Return"}
    border_color = "#AAB4C0"
    header_color = "#E6E6E6"
    label_fill = "#F5F5F5"
    text_color = "#202020"

    col_width_map = {
        "Year": 0.075,
        "Row Type": 0.125,
        "Annual Return": 0.095,
        "Win Rate": 0.075,
    }
    remaining_width = 1.0 - sum(col_width_map.values())
    month_width = remaining_width / 12.0
    normalized_widths = [col_width_map.get(column, month_width) for column in columns]

    x_positions = [left]
    running_x = left
    for col_width in normalized_widths[:-1]:
        running_x += width * col_width
        x_positions.append(running_x)

    header_ratio = 1.15
    total_units = header_ratio + len(summary_table)
    unit_height = height / total_units
    header_height = unit_height * header_ratio
    row_height = unit_height
    current_top = bottom + height

    heatmap_values = []
    for _, row_data in summary_table.iterrows():
        for column_name in columns:
            if column_name in return_columns:
                parsed_value = _parse_percent_string(row_data[column_name])
                if not np.isnan(parsed_value):
                    heatmap_values.append(parsed_value)
    max_abs_value = max((abs(value) for value in heatmap_values), default=0.0)

    for column_index, column_name in enumerate(columns):
        x0 = x_positions[column_index]
        cell_width = width * normalized_widths[column_index]
        ax.add_patch(
            Rectangle(
                (x0, current_top - header_height),
                cell_width,
                header_height,
                facecolor=header_color,
                edgecolor=border_color,
                linewidth=1.0,
            )
        )
        ax.text(
            x0 + cell_width / 2,
            current_top - header_height / 2,
            column_name,
            va="center",
            ha="center",
            fontsize=header_fontsize,
            fontweight="bold",
            color=text_color,
        )

    current_top -= header_height

    row_index = 0
    while row_index < len(summary_table):
        year_value = str(summary_table.iloc[row_index]["Year"])
        group_end = row_index + 1
        while group_end < len(summary_table) and summary_table.iloc[group_end]["Year"] == "":
            group_end += 1

        group_size = group_end - row_index
        merged_height = row_height * group_size

        year_x = x_positions[0]
        year_width = width * normalized_widths[0]
        ax.add_patch(
            Rectangle(
                (year_x, current_top - merged_height),
                year_width,
                merged_height,
                facecolor=label_fill,
                edgecolor=border_color,
                linewidth=0.9,
            )
        )
        ax.text(
            year_x + year_width / 2,
            current_top - merged_height / 2,
            year_value,
            va="center",
            ha="center",
            fontsize=body_fontsize,
            fontweight="bold",
            color=text_color,
        )

        row_top = current_top
        for inner_row in range(row_index, group_end):
            row_series = summary_table.iloc[inner_row]
            for column_index, column_name in enumerate(columns[1:], start=1):
                x0 = x_positions[column_index]
                cell_width = width * normalized_widths[column_index]
                cell_value = row_series[column_name]

                facecolor = "#FFFFFF"
                fontweight = "normal"
                if column_name in {"Row Type"}:
                    facecolor = label_fill
                    fontweight = "bold"
                elif column_name == "Win Rate":
                    fontweight = "bold"
                elif column_name in return_columns:
                    numeric_value = _parse_percent_string(cell_value)
                    facecolor = _heatmap_fill_color(numeric_value, max_abs_value)

                ax.add_patch(
                    Rectangle(
                        (x0, row_top - row_height),
                        cell_width,
                        row_height,
                        facecolor=facecolor,
                        edgecolor=border_color,
                        linewidth=0.85,
                    )
                )
                ax.text(
                    x0 + cell_width / 2,
                    row_top - row_height / 2,
                    str(cell_value),
                    va="center",
                    ha="center",
                    fontsize=body_fontsize,
                    fontweight=fontweight,
                    color=text_color,
                )

            row_top -= row_height

        current_top -= merged_height
        row_index = group_end


def _draw_metrics_summary_table(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    percentage_metrics: Optional[Iterable[str]] = None,
    output_path: str = f"{CHART_DIR}/metrics_summary_table.png",
) -> pd.DataFrame:
    """
    Draw a grouped metrics summary table with vertically merged category cells.
    """
    percentage_metrics = set(percentage_metrics or [])
    grouped_rows = _normalize_metric_categories(metric_categories)
    export_df = build_metrics_summary_table(
        metric_categories=metric_categories,
        percentage_metrics=percentage_metrics,
    )

    prepared_groups = []
    total_units = 1.0  # header row
    for category, rows in grouped_rows:
        prepared_rows = []
        for metric_name, metric_value in rows:
            metric_text = _display_metric_label(metric_name)
            value_text = _format_metric_value(metric_value, metric_name, percentage_metrics)
            metric_lines = _wrapped_lines(metric_text, width=28)
            value_lines = _wrapped_lines(value_text, width=34)
            line_count = max(len(metric_lines), len(value_lines))
            row_units = max(1.0, 0.9 + 0.45 * (line_count - 1))
            total_units += row_units
            prepared_rows.append(
                {
                    "metric_name": metric_name,
                    "metric_lines": metric_lines,
                    "value_lines": value_lines,
                    "row_units": row_units,
                }
            )

        prepared_groups.append({"category": category, "rows": prepared_rows})

    fig_height = max(4.5, total_units * 0.52)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    table_left = 0.03
    table_width = 0.94
    col_widths = [0.24, 0.41, 0.35]
    x_positions = [
        table_left,
        table_left + table_width * col_widths[0],
        table_left + table_width * (col_widths[0] + col_widths[1]),
    ]

    unit_height = 0.94 / total_units
    current_top = 0.97

    header_height = unit_height
    header_color = "#D8DEE9"
    border_color = "#AAB4C0"
    header_labels = ["Category", "Metric", "Value"]

    for column_index, header_label in enumerate(header_labels):
        x0 = x_positions[column_index]
        width = table_width * col_widths[column_index]
        ax.add_patch(
            Rectangle(
                (x0, current_top - header_height),
                width,
                header_height,
                facecolor=header_color,
                edgecolor=border_color,
                linewidth=1.1,
            )
        )
        ax.text(
            x0 + 0.012,
            current_top - header_height / 2,
            header_label,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#20262E",
        )

    current_top -= header_height
    category_fill = "#EEF2F7"
    cell_fill = "#FFFFFF"

    for group in prepared_groups:
        category_height = sum(row["row_units"] * unit_height for row in group["rows"])
        ax.add_patch(
            Rectangle(
                (x_positions[0], current_top - category_height),
                table_width * col_widths[0],
                category_height,
                facecolor=category_fill,
                edgecolor=border_color,
                linewidth=1.0,
            )
        )
        ax.text(
            x_positions[0] + 0.012,
            current_top - category_height / 2,
            group["category"],
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2933",
        )

        row_top = current_top
        for row in group["rows"]:
            row_height = row["row_units"] * unit_height
            for column_index, lines in ((1, row["metric_lines"]), (2, row["value_lines"])):
                x0 = x_positions[column_index]
                width = table_width * col_widths[column_index]
                ax.add_patch(
                    Rectangle(
                        (x0, row_top - row_height),
                        width,
                        row_height,
                        facecolor=cell_fill,
                        edgecolor=border_color,
                        linewidth=0.85,
                    )
                )
                ax.text(
                    x0 + 0.012,
                    row_top - row_height / 2,
                    "\n".join(lines),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#253240",
                    linespacing=1.35,
                )

            row_top -= row_height

        current_top -= category_height

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return export_df


def save_metrics_summary_table(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    percentage_metrics: Optional[Iterable[str]] = None,
    output_path: str = f"{CHART_DIR}/metrics_summary_table.png",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Save a grouped metrics summary table as PNG and CSV.
    """
    chart_dir, report_dir = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/metrics_summary_table.png":
        output_path = os.path.join(chart_dir, "metrics_summary_table.png")

    table_df = _draw_metrics_summary_table(
        metric_categories=metric_categories,
        percentage_metrics=percentage_metrics,
        output_path=output_path,
    )

    csv_path = os.path.join(report_dir, "metrics_summary_table.csv")
    table_df.to_csv(csv_path, index=False)
    return table_df


def save_metrics_summary_table_image(
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    output_path: str,
    percentage_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Save only the merged-cell metrics summary table image without writing the
    CSV companion export.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return _draw_metrics_summary_table(
        metric_categories=metric_categories,
        percentage_metrics=percentage_metrics,
        output_path=output_path,
    )


def plot_nav_and_drawdown(
    df: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/nav_drawdown.png",
    output_dir: Optional[str] = None,
    benchmark_name: str = "Benchmark",
) -> None:
    """
    Plot NAV as a line chart and drawdown as an area/line chart.
    """
    chart_dir, _ = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/nav_drawdown.png":
        output_path = os.path.join(chart_dir, "nav_drawdown.png")

    x_axis = _date_axis(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True)

    normalized_fund = df["nav"] / df["nav"].iloc[0]
    axes[0].plot(x_axis, normalized_fund, label="Fund NAV", color="#1f77b4", linewidth=1.8)
    if "benchmark" in df.columns:
        normalized_benchmark = df["benchmark"] / df["benchmark"].iloc[0]
        axes[0].plot(x_axis, normalized_benchmark, label=benchmark_name, color="#ff7f0e", linewidth=1.4)
    excess_axis = axes[0].twinx()
    # Relative growth ratio minus one is used here as cumulative benchmark-relative performance.
    cumulative_excess = (
        (1 + pd.to_numeric(df["fund_return"], errors="coerce")).fillna(1.0).cumprod()
        / (1 + pd.to_numeric(df["benchmark_return"], errors="coerce")).fillna(1.0).cumprod()
        - 1
    )
    excess_axis.plot(
        x_axis,
        cumulative_excess,
        label=f"Cumulative Excess Return vs {benchmark_name}",
        color="#2E7D32",
        linewidth=1.4,
        linestyle="--",
    )
    axes[0].set_title(f"Fund NAV vs {benchmark_name}", pad=26)
    axes[0].set_ylabel("Normalized Level (Base = 1.0)")
    excess_axis.set_ylabel("Cumulative Excess Return")
    excess_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axes[0].text(
        0.01,
        1.01,
        f"Rebased to 1.0 at inception; {benchmark_name} aligned backward to fund NAV dates",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#55606E",
        va="bottom",
    )
    left_handles, left_labels = axes[0].get_legend_handles_labels()
    right_handles, right_labels = excess_axis.get_legend_handles_labels()
    axes[0].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.93),
        frameon=True,
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_axis, df["drawdown"], label="Drawdown", color="#d62728", linewidth=1.5)
    axes[1].fill_between(x_axis, df["drawdown"], 0, color="#d62728", alpha=0.25)
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.88, hspace=0.26)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_metrics(
    df: pd.DataFrame,
    output_path: str = f"{CHART_DIR}/rolling_metrics.png",
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot rolling volatility and rolling Sharpe ratio as line charts.
    """
    chart_dir, _ = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/rolling_metrics.png":
        output_path = os.path.join(chart_dir, "rolling_metrics.png")

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
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot monthly returns as a bar chart.
    """
    chart_dir, _ = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/monthly_returns.png":
        output_path = os.path.join(chart_dir, "monthly_returns.png")

    if monthly_return_table.empty:
        return

    data = monthly_return_table.copy()
    colors = np.where(data["monthly_return"] >= 0, "#C62828", "#2E7D32")

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
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot annual returns as a bar chart.
    """
    chart_dir, _ = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/annual_returns.png":
        output_path = os.path.join(chart_dir, "annual_returns.png")

    if annual_return_table.empty:
        return

    data = annual_return_table.copy()
    data["year"] = data["year"].astype(str)
    colors = np.where(data["annual_return"] >= 0, "#C62828", "#2E7D32")

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
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot drawdown frequency tables as bar charts.
    """
    chart_dir, _ = _ensure_output_dirs(output_dir=output_dir)
    if output_path == f"{CHART_DIR}/drawdown_frequency.png":
        output_path = os.path.join(chart_dir, "drawdown_frequency.png")

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
    metric_categories: Dict[str, Dict[str, object] | Sequence[Tuple[str, object]]],
    monthly_return_table: pd.DataFrame,
    benchmark_monthly_return_table: pd.DataFrame,
    monthly_excess_return_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_annual_return_table: pd.DataFrame,
    drawdown_frequencies: Dict[str, pd.DataFrame],
    percentage_metrics: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
    benchmark_name: str = "Benchmark",
) -> pd.DataFrame:
    """
    Generate all analysis tables and charts.
    """
    metrics_table = save_metrics_summary_table(
        metric_categories,
        percentage_metrics=percentage_metrics,
        output_dir=output_dir,
    )
    plot_nav_and_drawdown(df, output_dir=output_dir, benchmark_name=benchmark_name)
    plot_rolling_metrics(df, output_dir=output_dir)
    plot_monthly_returns(monthly_return_table, output_dir=output_dir)
    plot_annual_returns(annual_return_table, output_dir=output_dir)
    plot_drawdown_frequency(drawdown_frequencies, output_dir=output_dir)

    return metrics_table
