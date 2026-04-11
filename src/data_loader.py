import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.frequency import infer_data_frequency, is_supported_frequency
from src.relative_performance import calculate_period_excess_returns


def _read_timeseries_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV or Excel time-series file into a DataFrame.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file type for '{file_path}'. Supported types are CSV, XLSX, and XLS."
    )


def load_two_column_timeseries(file_path: str, value_column_name: str) -> pd.DataFrame:
    """
    Load a generic two-column time series and standardize it to date + value columns.

    Assumptions:
    - The first column contains dates.
    - The second column contains a level series.
    - For benchmark inputs, the second column must be benchmark levels such as
      benchmark NAV, index level, or close price. It must not be a return series,
      because the pipeline calculates pct_change() after alignment.
    """
    raw_df = _read_timeseries_file(file_path)
    if raw_df.shape[1] < 2:
        raise ValueError(
            f"File '{file_path}' must contain at least two columns: date and {value_column_name}."
        )

    df = raw_df.iloc[:, :2].copy()
    df.columns = ["date", value_column_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_column_name] = pd.to_numeric(df[value_column_name], errors="coerce")
    df = df.dropna(subset=["date", value_column_name])

    if df.empty:
        raise ValueError(
            f"File '{file_path}' does not contain any valid date/{value_column_name} rows after cleaning."
        )

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def load_nav_data(file_path: str) -> pd.DataFrame:
    """
    Load fund NAV data and standardize columns to ['date', 'nav'].
    """
    return load_two_column_timeseries(file_path, value_column_name="nav")


def load_benchmark_data(file_path: str) -> pd.DataFrame:
    """
    Load benchmark level data and standardize columns to ['date', 'benchmark'].

    Important validation note:
    - The benchmark file's second column must be a benchmark level series.
    - Do not provide already-calculated benchmark returns in this file.
    """
    return load_two_column_timeseries(file_path, value_column_name="benchmark")


def derive_benchmark_name(file_path: str) -> str:
    """
    Derive a readable benchmark name from the input file name when no explicit
    benchmark label is provided by the CLI.
    """
    stem = Path(file_path).stem
    stem = re.sub(r"^benchmark[_\-\s]*", "", stem, flags=re.IGNORECASE).strip()
    stem = re.sub(r"[_\-]+", " ", stem).strip()

    return stem or "Benchmark"


def infer_dataset_frequency(
    df: pd.DataFrame,
    dataset_name: str,
    require_supported_frequency: bool = True,
) -> Tuple[str, float]:
    """
    Infer dataset frequency and raise a clean error for unsupported or weak inputs.
    """
    frequency, periods_per_year = infer_data_frequency(df, strict=True)

    if require_supported_frequency and not is_supported_frequency(frequency):
        raise ValueError(
            f"Unsupported {dataset_name} frequency '{frequency}'. "
            "Supported frequencies are daily, weekly, and monthly."
        )

    return frequency, periods_per_year


def align_fund_and_benchmark(
    fund_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Align benchmark levels onto the fund NAV date schedule.

    Rules:
    - Fund dates define the master analysis timeline.
    - The benchmark can be higher frequency than the fund.
    - Each fund date uses the latest benchmark value on or before that date.
    - Future benchmark values are never used.
    - A lower-frequency benchmark is rejected because it cannot reliably support
      the higher-frequency fund timeline.
    """
    fund_frequency, fund_periods_per_year = infer_dataset_frequency(fund_df, "fund")
    benchmark_frequency, benchmark_periods_per_year = infer_dataset_frequency(
        benchmark_df,
        "benchmark",
        require_supported_frequency=False,
    )

    if benchmark_periods_per_year < fund_periods_per_year:
        raise ValueError(
            "Benchmark data is lower frequency than the fund NAV series and cannot be aligned "
            "reliably to the fund analysis timeline. "
            f"Fund frequency: {fund_frequency}. Benchmark frequency: {benchmark_frequency}."
        )

    aligned = pd.merge_asof(
        fund_df.sort_values("date"),
        benchmark_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    print(
        "WARNING: benchmark is aligned via merge_asof(direction='backward') — "
        "equivalent to forward-fill. This may introduce bias if the benchmark "
        "has significantly fewer observations than the fund."
    )
    unmatched_rows = int(aligned["benchmark"].isna().sum())
    aligned = aligned.dropna(subset=["benchmark"]).reset_index(drop=True)

    if aligned.empty:
        raise ValueError(
            "No aligned overlapping observations were found between the fund dates and the benchmark history."
        )

    aligned = aligned.sort_values("date").reset_index(drop=True)
    aligned["fund_return"] = aligned["nav"].pct_change()
    aligned["benchmark_return"] = aligned["benchmark"].pct_change()

    fund_ret_clean = aligned["fund_return"].dropna()
    bench_ret_clean = aligned["benchmark_return"].dropna()
    if not fund_ret_clean.index.equals(bench_ret_clean.index):
        raise AssertionError(
            "Fund and benchmark return indices diverge after alignment. "
            f"Fund return rows: {len(fund_ret_clean)}, Benchmark return rows: {len(bench_ret_clean)}"
        )

    aligned["period_excess_return"] = calculate_period_excess_returns(
        aligned["fund_return"],
        aligned["benchmark_return"],
    ).reindex(aligned.index)

    aligned_return_rows = aligned[["fund_return", "benchmark_return"]].dropna()
    if not aligned_return_rows.index.equals(aligned.loc[aligned_return_rows.index].index):
        raise AssertionError("Aligned fund and benchmark return rows must preserve a shared index.")

    metadata = {
        "fund_frequency": fund_frequency,
        "benchmark_frequency": benchmark_frequency,
        "frequency": fund_frequency,
        "periods_per_year": fund_periods_per_year,
        "benchmark_periods_per_year": benchmark_periods_per_year,
        "overlapping_observations": len(aligned),
        "dropped_unmatched_fund_rows": unmatched_rows,
        "aligned_start_date": aligned["date"].min(),
        "aligned_end_date": aligned["date"].max(),
        "alignment_method": "benchmark mapped backward onto fund dates",
    }

    return aligned, metadata


def load_and_align_datasets(
    fund_file_path: str,
    benchmark_file_path: str,
    benchmark_name: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load, standardize, validate, and align fund and benchmark time series.
    """
    fund_df = load_nav_data(fund_file_path)
    benchmark_df = load_benchmark_data(benchmark_file_path)
    aligned_df, metadata = align_fund_and_benchmark(fund_df, benchmark_df)
    metadata["benchmark_name"] = benchmark_name or derive_benchmark_name(benchmark_file_path)
    return aligned_df, metadata
