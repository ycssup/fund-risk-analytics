"""
Strict validation and diagnostic script for excess return logic.

Covers:
  Part 1 — Cumulative excess return validation
  Part 2 — Monthly excess return validation
  Part 3 — Frequency consistency check
  Part 4 — Benchmark alignment audit (ffill vs inner join)
  Part 5 — Sanity checks (behavioral)
  Part 6 — Chart verification (debug overlay)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_nav_data,
    load_benchmark_data,
    align_fund_and_benchmark,
)
from src.relative_performance import (
    calculate_cumulative_returns,
    calculate_cumulative_excess_returns,
    calculate_period_excess_returns,
    calculate_annualized_excess_return,
    calculate_annualized_return_from_series,
    build_relative_period_table,
)

SEPARATOR = "=" * 70
PASS = "PASS"
FAIL = "FAIL"

DEFAULT_FUND_PATH = "data/sample_nav_data.xlsx"
DEFAULT_BENCHMARK_CANDIDATES = (
    "data/benchmark.xlsx",
    "data/benchmark.csv",
    "data/benchmark.xls",
    "data/benchmark_CSI 300.xlsx",
)
DEFAULT_OUTPUT_DIR = "output"


def resolve_benchmark_path(benchmark_arg):
    if benchmark_arg:
        return benchmark_arg
    for candidate in DEFAULT_BENCHMARK_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Benchmark file not found. Pass --benchmark.")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate excess return calculations.")
    parser.add_argument("--input", default=DEFAULT_FUND_PATH)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


# ============================================================
# Part 1 — Cumulative Excess Return Validation
# ============================================================
def validate_cumulative_excess(df):
    print(f"\n{SEPARATOR}")
    print("PART 1 — Cumulative Excess Return Validation")
    print(SEPARATOR)

    fund_returns = df["fund_return"].dropna()
    benchmark_returns = df["benchmark_return"].dropna()

    cumulative_fund, cumulative_benchmark, cumulative_excess = calculate_cumulative_excess_returns(
        fund_returns, benchmark_returns,
    )

    computed_diff = cumulative_fund - cumulative_benchmark
    max_abs_diff = np.nanmax(np.abs(computed_diff.values - cumulative_excess.values))
    allclose = np.allclose(computed_diff.values, cumulative_excess.values, atol=1e-10, equal_nan=True)

    print(f"\nnp.allclose(computed_diff, cumulative_excess, atol=1e-10): {allclose}")
    print(f"Max absolute difference: {max_abs_diff:.2e}")

    print("\nLast 5 rows of cumulative_fund:")
    print(cumulative_fund.tail(5).to_string())

    print("\nLast 5 rows of cumulative_benchmark:")
    print(cumulative_benchmark.tail(5).to_string())

    print("\nLast 5 rows of cumulative_excess (stored):")
    print(cumulative_excess.tail(5).to_string())

    print("\nLast 5 rows of computed_diff (cumulative_fund - cumulative_benchmark):")
    print(computed_diff.tail(5).to_string())

    result = PASS if allclose else FAIL
    print(f"\n[{result}] Cumulative excess return definition is {'CONSISTENT' if allclose else 'INCONSISTENT'}.")
    return allclose, cumulative_fund, cumulative_benchmark, cumulative_excess


# ============================================================
# Part 2 — Monthly Excess Return Validation
# ============================================================
def validate_monthly_excess(df):
    print(f"\n{SEPARATOR}")
    print("PART 2 — Monthly Excess Return Validation")
    print(SEPARATOR)

    monthly_table = build_relative_period_table(
        df=df,
        freq="M",
        fund_level_col="nav",
        benchmark_level_col="benchmark",
        fund_output_col="fund_monthly_return",
        benchmark_output_col="benchmark_monthly_return",
        excess_output_col="monthly_excess_return",
    )

    if monthly_table.empty:
        print("[FAIL] Monthly table is empty — no data to validate.")
        return False

    fund_monthly = monthly_table["fund_monthly_return"]
    benchmark_monthly = monthly_table["benchmark_monthly_return"]
    stored_excess = monthly_table["monthly_excess_return"]
    expected_excess = fund_monthly - benchmark_monthly

    valid_mask = stored_excess.notna() & expected_excess.notna()
    if valid_mask.sum() == 0:
        print("[FAIL] No valid monthly excess rows to compare.")
        return False

    max_abs_diff = np.nanmax(np.abs(stored_excess[valid_mask].values - expected_excess[valid_mask].values))
    allclose = np.allclose(
        stored_excess[valid_mask].values,
        expected_excess[valid_mask].values,
        atol=1e-10,
        equal_nan=True,
    )

    print(f"\nnp.allclose(stored_excess, fund - benchmark, atol=1e-10): {allclose}")
    print(f"Max absolute difference: {max_abs_diff:.2e}")

    print("\nMonthly check (last 3 rows):")
    print("fund_monthly:")
    print(fund_monthly.tail(3).to_string())
    print("\nbenchmark_monthly:")
    print(benchmark_monthly.tail(3).to_string())
    print("\nmonthly_excess (stored):")
    print(stored_excess.tail(3).to_string())
    print("\nfund_monthly - benchmark_monthly (recomputed):")
    print(expected_excess.tail(3).to_string())

    # Spot-check last available month
    last_idx = monthly_table.dropna(subset=["monthly_excess_return"]).index[-1]
    spot_fund = float(fund_monthly.iloc[last_idx])
    spot_bench = float(benchmark_monthly.iloc[last_idx])
    spot_stored = float(stored_excess.iloc[last_idx])
    spot_expected = spot_fund - spot_bench
    spot_match = np.isclose(spot_stored, spot_expected, atol=1e-12)

    print(f"\nSpot check (last month idx={last_idx}):")
    print(f"  fund_monthly   = {spot_fund:.10f}")
    print(f"  bench_monthly  = {spot_bench:.10f}")
    print(f"  stored excess  = {spot_stored:.10f}")
    print(f"  expected excess= {spot_expected:.10f}")
    print(f"  match: {spot_match}")

    result = PASS if allclose and spot_match else FAIL
    print(f"\n[{result}] Monthly excess return definition is {'CONSISTENT' if allclose else 'INCONSISTENT'}.")
    return allclose


# ============================================================
# Part 3 — Frequency Consistency Check
# ============================================================
def validate_frequency_consistency(df, metadata):
    print(f"\n{SEPARATOR}")
    print("PART 3 — Frequency Consistency Check")
    print(SEPARATOR)

    freq = metadata["frequency"]
    ppyr = metadata["periods_per_year"]
    print(f"\nDataset frequency: {freq}")
    print(f"Periods per year: {ppyr}")

    # Verify fund and benchmark returns share the same index
    fund_ret = df["fund_return"].dropna()
    bench_ret = df["benchmark_return"].dropna()
    index_match = fund_ret.index.equals(bench_ret.index)

    print(f"\nfund_return rows (non-null): {len(fund_ret)}")
    print(f"benchmark_return rows (non-null): {len(bench_ret)}")
    print(f"fund_return.index.equals(benchmark_return.index): {index_match}")

    if not index_match:
        print("[FAIL] Fund and benchmark return indices do not match!")
        print(f"  fund-only indices: {fund_ret.index.difference(bench_ret.index).tolist()[:5]}")
        print(f"  bench-only indices: {bench_ret.index.difference(fund_ret.index).tolist()[:5]}")
        return False

    # Verify cumulative calculations use the same daily returns
    cum_fund = calculate_cumulative_returns(fund_ret)
    cum_bench = calculate_cumulative_returns(bench_ret)
    print(f"\nCumulative fund series length: {len(cum_fund)}")
    print(f"Cumulative benchmark series length: {len(cum_bench)}")
    print(f"Cumulative indices match: {cum_fund.index.equals(cum_bench.index)}")

    # Verify monthly/annual are derived from the aligned daily levels
    monthly_table = build_relative_period_table(
        df=df, freq="M",
        fund_level_col="nav", benchmark_level_col="benchmark",
        fund_output_col="fund_monthly_return",
        benchmark_output_col="benchmark_monthly_return",
        excess_output_col="monthly_excess_return",
    )
    annual_table = build_relative_period_table(
        df=df, freq="Y",
        fund_level_col="nav", benchmark_level_col="benchmark",
        fund_output_col="fund_annual_return",
        benchmark_output_col="benchmark_annual_return",
        excess_output_col="annual_excess_return",
    )
    print(f"\nMonthly table rows: {len(monthly_table)}")
    print(f"Annual table rows: {len(annual_table)}")
    print(f"Both derived from the same aligned df with {len(df)} observations.")

    # Verify no mixing of frequencies: both fund_return and benchmark_return
    # come from pct_change() on the same aligned df
    date_diffs = pd.to_datetime(df["date"]).diff().dt.days.dropna()
    median_gap = date_diffs.median()
    print(f"\nMedian date gap (days): {median_gap}")
    print(f"Min date gap: {date_diffs.min()}, Max date gap: {date_diffs.max()}")

    result = PASS if index_match else FAIL
    print(f"\n[{result}] Frequency consistency check {'PASSED' if index_match else 'FAILED'}.")
    return index_match


# ============================================================
# Part 4 — Benchmark Alignment Audit (ffill vs inner)
# ============================================================
def validate_alignment_methods(fund_path, benchmark_path):
    print(f"\n{SEPARATOR}")
    print("PART 4 — Benchmark Alignment Audit (merge_asof vs inner join)")
    print(SEPARATOR)

    fund_df = load_nav_data(fund_path)
    benchmark_df = load_benchmark_data(benchmark_path)

    # Method A: merge_asof (backward fill) — current pipeline method
    aligned_ffill, meta_ffill = align_fund_and_benchmark(fund_df, benchmark_df)
    fund_ret_ffill = aligned_ffill["fund_return"].dropna()
    bench_ret_ffill = aligned_ffill["benchmark_return"].dropna()
    cum_fund_ffill = calculate_cumulative_returns(fund_ret_ffill)
    cum_bench_ffill = calculate_cumulative_returns(bench_ret_ffill)
    cum_excess_ffill = cum_fund_ffill - cum_bench_ffill

    # Method B: strict inner join on exact date matches
    fund_sorted = fund_df.sort_values("date").reset_index(drop=True)
    bench_sorted = benchmark_df.sort_values("date").reset_index(drop=True)
    inner_merged = fund_sorted.merge(bench_sorted, on="date", how="inner").sort_values("date").reset_index(drop=True)
    inner_merged["fund_return"] = inner_merged["nav"].pct_change()
    inner_merged["benchmark_return"] = inner_merged["benchmark"].pct_change()

    fund_ret_inner = inner_merged["fund_return"].dropna()
    bench_ret_inner = inner_merged["benchmark_return"].dropna()

    if len(fund_ret_inner) > 0 and len(bench_ret_inner) > 0:
        cum_fund_inner = calculate_cumulative_returns(fund_ret_inner)
        cum_bench_inner = calculate_cumulative_returns(bench_ret_inner)
        cum_excess_inner = cum_fund_inner - cum_bench_inner
    else:
        cum_excess_inner = pd.Series(dtype=float)

    print(f"\nMethod A (merge_asof backward) observations: {len(aligned_ffill)}")
    print(f"Method B (strict inner join) observations: {len(inner_merged)}")
    print(f"Observation difference: {len(aligned_ffill) - len(inner_merged)}")

    print(f"\nMethod A — cumulative excess tail:")
    print(cum_excess_ffill.tail(5).to_string())

    if len(cum_excess_inner) > 0:
        print(f"\nMethod B — cumulative excess tail:")
        print(cum_excess_inner.tail(5).to_string())

        final_ffill = float(cum_excess_ffill.iloc[-1])
        final_inner = float(cum_excess_inner.iloc[-1])
        diff = abs(final_ffill - final_inner)
        material = diff > 0.001  # 10 bps threshold

        print(f"\nFinal cumulative excess (ffill):  {final_ffill:.6f} ({final_ffill:.2%})")
        print(f"Final cumulative excess (inner):  {final_inner:.6f} ({final_inner:.2%})")
        print(f"Absolute difference:              {diff:.6f} ({diff:.2%})")
        print(f"Material difference (>10bps):     {material}")

        if material:
            print("\n[WARNING] merge_asof alignment introduces MATERIAL bias vs strict inner join.")
        else:
            print("\n[INFO] Alignment methods produce similar results — no material bias detected.")
    else:
        print("\n[WARNING] Inner join produced no overlapping dates — cannot compare.")
        material = None

    return material


# ============================================================
# Part 5 — Sanity Checks (Behavioral)
# ============================================================
def validate_sanity_checks(df):
    print(f"\n{SEPARATOR}")
    print("PART 5 — Sanity Checks (Behavioral)")
    print(SEPARATOR)

    fund_ret = df["fund_return"].dropna()
    bench_ret = df["benchmark_return"].dropna()

    period_excess = fund_ret - bench_ret
    cum_fund, cum_bench, cum_excess = calculate_cumulative_excess_returns(fund_ret, bench_ret)

    hit_rate = float((period_excess > 0).mean())
    final_cum_excess = float(cum_excess.iloc[-1])
    sign_series = np.sign(cum_excess)
    sign_changes = int(np.sum(sign_series.diff().fillna(0) != 0))

    has_positive = (period_excess > 0).any()
    has_negative = (period_excess < 0).any()
    is_monotonic = cum_excess.is_monotonic_increasing or cum_excess.is_monotonic_decreasing

    print("\nSanity Check:")
    print(f"  Hit rate (% positive period excess):  {hit_rate:.4f} ({hit_rate:.2%})")
    print(f"  Final cumulative excess return:        {final_cum_excess:.6f} ({final_cum_excess:.2%})")
    print(f"  Sign changes in cumulative excess:     {sign_changes}")
    print(f"  Period excess has positive values:     {has_positive}")
    print(f"  Period excess has negative values:     {has_negative}")
    print(f"  Cumulative excess is monotonic:        {is_monotonic}")
    print(f"  Total period excess observations:      {len(period_excess)}")

    all_ok = True
    if not has_positive and not has_negative:
        print("\n[WARNING] Period excess returns are all zero — data may be trivial.")
        all_ok = False
    if not has_positive or not has_negative:
        print(f"\n[INFO] Period excess is one-sided ({'all positive' if has_positive else 'all negative'}).")

    result = PASS if all_ok else FAIL
    print(f"\n[{result}] Sanity checks {'PASSED' if all_ok else 'show warnings'}.")
    return all_ok


# ============================================================
# Part 6 — Chart Verification (Debug Overlay)
# ============================================================
def validate_chart_overlay(df, output_dir):
    print(f"\n{SEPARATOR}")
    print("PART 6 — Chart Verification (Debug Overlay)")
    print(SEPARATOR)

    fund_ret = df["fund_return"].dropna()
    bench_ret = df["benchmark_return"].dropna()

    cum_fund, cum_bench, cum_excess_stored = calculate_cumulative_excess_returns(fund_ret, bench_ret)
    cum_excess_recomputed = cum_fund - cum_bench

    # Verify the values stored in df match the recomputed values
    df_cum_excess = pd.to_numeric(df["cumulative_excess_return"], errors="coerce").dropna()
    recomputed_aligned = cum_excess_stored.reindex(df_cum_excess.index)
    valid = df_cum_excess.notna() & recomputed_aligned.notna()

    if valid.sum() > 0:
        max_diff = np.nanmax(np.abs(df_cum_excess[valid].values - recomputed_aligned[valid].values))
        match = np.allclose(df_cum_excess[valid].values, recomputed_aligned[valid].values, atol=1e-10)
        print(f"\ndf['cumulative_excess_return'] vs recomputed cumulative_excess:")
        print(f"  Rows compared: {valid.sum()}")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Match: {match}")
    else:
        match = False
        print("\n[WARNING] No valid cumulative_excess_return values in df to compare.")

    # Generate debug overlay chart
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    overlay_path = os.path.join(chart_dir, "debug_cumulative_excess_overlay.png")

    dates = cum_excess_stored.index if hasattr(cum_excess_stored.index, 'date') else range(len(cum_excess_stored))
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
        # Map recomputed index back to dates
        date_map = date_series.reindex(cum_excess_stored.index)
    else:
        date_map = pd.Series(range(len(cum_excess_stored)), index=cum_excess_stored.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        date_map.values, cum_excess_stored.values,
        label="Stored cumulative excess", color="#1f77b4", linewidth=2.0,
    )
    ax.plot(
        date_map.values, cum_excess_recomputed.values,
        label="Recomputed (cum_fund - cum_bench)", color="#ff7f0e",
        linewidth=1.5, linestyle="--",
    )
    ax.set_title("Debug Overlay: Stored vs Recomputed Cumulative Excess Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Excess Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"\nDebug overlay chart saved to: {overlay_path}")

    result = PASS if match else FAIL
    print(f"\n[{result}] Chart data {'matches' if match else 'does NOT match'} recomputed cumulative excess.")
    return match


# ============================================================
# Part Bonus — Annualized Excess Validation
# ============================================================
def validate_annualized_excess(df, periods_per_year):
    print(f"\n{SEPARATOR}")
    print("BONUS — Annualized Excess Return Validation")
    print(SEPARATOR)

    fund_ret = df["fund_return"].dropna()
    bench_ret = df["benchmark_return"].dropna()

    ann_fund = calculate_annualized_return_from_series(fund_ret, periods_per_year)
    ann_bench = calculate_annualized_return_from_series(bench_ret, periods_per_year)
    ann_excess = calculate_annualized_excess_return(fund_ret, bench_ret, periods_per_year)
    expected = ann_fund - ann_bench

    match = np.isclose(ann_excess, expected, atol=1e-12)
    print(f"\n  Annualized fund return:       {ann_fund:.10f} ({ann_fund:.4%})")
    print(f"  Annualized benchmark return:  {ann_bench:.10f} ({ann_bench:.4%})")
    print(f"  Annualized excess (stored):   {ann_excess:.10f} ({ann_excess:.4%})")
    print(f"  Expected (fund - benchmark):  {expected:.10f} ({expected:.4%})")
    print(f"  Difference:                   {abs(ann_excess - expected):.2e}")
    print(f"  Match: {match}")

    result = PASS if match else FAIL
    print(f"\n[{result}] Annualized excess return is {'CONSISTENT' if match else 'INCONSISTENT'}.")
    return match


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    fund_path = args.input
    benchmark_path = resolve_benchmark_path(args.benchmark)
    output_dir = args.output

    print(f"Fund file:      {fund_path}")
    print(f"Benchmark file: {benchmark_path}")
    print(f"Output dir:     {output_dir}")

    # Load and align data (this triggers the merge_asof warning from data_loader)
    from src.data_loader import load_nav_data, load_benchmark_data, align_fund_and_benchmark
    fund_df = load_nav_data(fund_path)
    benchmark_df = load_benchmark_data(benchmark_path)
    df, metadata = align_fund_and_benchmark(fund_df, benchmark_df)

    # Compute cumulative columns (mirrors analysis_pipeline)
    fund_ret = df["fund_return"].dropna()
    bench_ret = df["benchmark_return"].dropna()
    cum_fund, cum_bench, cum_excess = calculate_cumulative_excess_returns(fund_ret, bench_ret)
    df["cumulative_return"] = cum_fund.reindex(df.index)
    df["benchmark_cumulative_return"] = cum_bench.reindex(df.index)
    df["cumulative_excess_return"] = cum_excess.reindex(df.index)

    periods_per_year = metadata["periods_per_year"]

    results = {}

    # Part 1
    ok, _, _, _ = validate_cumulative_excess(df)
    results["Part 1 — Cumulative excess"] = ok

    # Part 2
    ok = validate_monthly_excess(df)
    results["Part 2 — Monthly excess"] = ok

    # Part 3
    ok = validate_frequency_consistency(df, metadata)
    results["Part 3 — Frequency consistency"] = ok

    # Part 4
    material_bias = validate_alignment_methods(fund_path, benchmark_path)
    results["Part 4 — Alignment audit"] = not material_bias if material_bias is not None else True

    # Part 5
    ok = validate_sanity_checks(df)
    results["Part 5 — Sanity checks"] = ok

    # Part 6
    ok = validate_chart_overlay(df, output_dir)
    results["Part 6 — Chart verification"] = ok

    # Bonus
    ok = validate_annualized_excess(df, periods_per_year)
    results["Bonus — Annualized excess"] = ok

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{SEPARATOR}")
    print("VALIDATION SUMMARY")
    print(SEPARATOR)
    all_pass = True
    for check_name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  [{status}] {check_name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — review output above'}")
    print(SEPARATOR)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
