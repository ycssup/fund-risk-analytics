import math
import unittest

import numpy as np
import pandas as pd

from src.relative_performance import (
    build_relative_period_table,
    calculate_annualized_excess_return,
    calculate_cumulative_excess_returns,
    calculate_hit_rate,
    calculate_period_excess_returns,
)
from src.risk_adjusted_return import benchmark_comparison_metrics
from src.visualization import build_monthly_returns_heatmap_table


class RelativePerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monthly_dates = pd.to_datetime(
            ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]
        )
        self.fund_returns = pd.Series([0.10, -0.05, 0.03, 0.04], index=self.monthly_dates)
        self.benchmark_returns = pd.Series([0.08, -0.02, 0.01, 0.02], index=self.monthly_dates)

        fund_levels = [100.0]
        benchmark_levels = [100.0]
        for fund_return, benchmark_return in zip(self.fund_returns, self.benchmark_returns):
            fund_levels.append(fund_levels[-1] * (1 + fund_return))
            benchmark_levels.append(benchmark_levels[-1] * (1 + benchmark_return))

        self.aligned_df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]
                ),
                "nav": fund_levels,
                "benchmark": benchmark_levels,
            }
        )
        self.aligned_df["fund_return"] = self.aligned_df["nav"].pct_change()
        self.aligned_df["benchmark_return"] = self.aligned_df["benchmark"].pct_change()
        self.aligned_df["period_excess_return"] = (
            self.aligned_df["fund_return"] - self.aligned_df["benchmark_return"]
        )

    def test_period_excess_returns_use_simple_period_subtraction(self) -> None:
        period_excess = calculate_period_excess_returns(self.fund_returns, self.benchmark_returns)
        expected = pd.Series([0.02, -0.03, 0.02, 0.02], index=self.monthly_dates, name="period_excess_return")
        pd.testing.assert_series_equal(period_excess, expected)

    def test_cumulative_excess_return_is_difference_between_cumulative_series(self) -> None:
        cumulative_fund, cumulative_benchmark, cumulative_excess = calculate_cumulative_excess_returns(
            self.fund_returns,
            self.benchmark_returns,
        )
        expected_cumulative_excess = (cumulative_fund - cumulative_benchmark).rename("cumulative_excess_return")
        pd.testing.assert_series_equal(cumulative_excess, expected_cumulative_excess, check_names=True)

        compounded_period_excess = (1 + calculate_period_excess_returns(self.fund_returns, self.benchmark_returns)).cumprod() - 1
        self.assertFalse(np.allclose(cumulative_excess.values, compounded_period_excess.values))

    def test_annualized_excess_return_is_difference_between_annualized_returns(self) -> None:
        annualized_excess_return = calculate_annualized_excess_return(
            self.fund_returns,
            self.benchmark_returns,
            periods_per_year=12,
        )
        expected_fund = float((1 + self.fund_returns).prod() ** (12 / len(self.fund_returns)) - 1)
        expected_benchmark = float((1 + self.benchmark_returns).prod() ** (12 / len(self.benchmark_returns)) - 1)
        self.assertAlmostEqual(annualized_excess_return, expected_fund - expected_benchmark, places=12)

    def test_hit_rate_uses_positive_period_excess(self) -> None:
        period_excess = calculate_period_excess_returns(self.fund_returns, self.benchmark_returns)
        self.assertAlmostEqual(calculate_hit_rate(period_excess), 0.75, places=12)

    def test_relative_period_tables_produce_explicit_monthly_and_annual_excess_returns(self) -> None:
        monthly_table = build_relative_period_table(
            df=self.aligned_df,
            freq="M",
            fund_level_col="nav",
            benchmark_level_col="benchmark",
            fund_output_col="fund_monthly_return",
            benchmark_output_col="benchmark_monthly_return",
            excess_output_col="monthly_excess_return",
        )
        self.assertTrue(
            np.allclose(monthly_table["monthly_excess_return"].dropna().tail(4), [0.02, -0.03, 0.02, 0.02])
        )

        annual_table = build_relative_period_table(
            df=self.aligned_df,
            freq="Y",
            fund_level_col="nav",
            benchmark_level_col="benchmark",
            fund_output_col="fund_annual_return",
            benchmark_output_col="benchmark_annual_return",
            excess_output_col="annual_excess_return",
        )
        self.assertAlmostEqual(
            annual_table.loc[0, "annual_excess_return"],
            annual_table.loc[0, "fund_annual_return"] - annual_table.loc[0, "benchmark_annual_return"],
            places=12,
        )

    def test_heatmap_table_uses_explicit_row_labels_and_win_rate(self) -> None:
        monthly_table = build_relative_period_table(
            df=self.aligned_df,
            freq="M",
            fund_level_col="nav",
            benchmark_level_col="benchmark",
            fund_output_col="fund_monthly_return",
            benchmark_output_col="benchmark_monthly_return",
            excess_output_col="monthly_excess_return",
        )
        annual_table = build_relative_period_table(
            df=self.aligned_df,
            freq="Y",
            fund_level_col="nav",
            benchmark_level_col="benchmark",
            fund_output_col="fund_annual_return",
            benchmark_output_col="benchmark_annual_return",
            excess_output_col="annual_excess_return",
        )
        summary = build_monthly_returns_heatmap_table(
            monthly_return_table=monthly_table[["month", "fund_monthly_return"]].rename(
                columns={"fund_monthly_return": "monthly_return"}
            ),
            benchmark_monthly_return_table=monthly_table[["month", "benchmark_monthly_return"]],
            monthly_excess_return_table=monthly_table,
            annual_return_table=annual_table[["year", "fund_annual_return"]].rename(
                columns={"fund_annual_return": "annual_return"}
            ),
            benchmark_annual_return_table=annual_table[["year", "benchmark_annual_return"]],
            annual_excess_return_table=annual_table,
        )
        self.assertEqual(summary.loc[0, "Row Type"], "Fund Monthly Return")
        self.assertEqual(summary.loc[1, "Row Type"], "Benchmark Monthly Return")
        self.assertEqual(summary.loc[2, "Row Type"], "Monthly Excess Return")

        monthly_excess_2024_row = summary.iloc[5]
        self.assertEqual(monthly_excess_2024_row["Row Type"], "Monthly Excess Return")
        self.assertEqual(monthly_excess_2024_row["Win Rate"], "75.00%")

    def test_benchmark_comparison_metrics_expose_explicit_relative_fields(self) -> None:
        metrics = benchmark_comparison_metrics(self.aligned_df, periods_per_year=12)
        self.assertIn("benchmark_annualized_return", metrics)
        self.assertIn("annualized_excess_return", metrics)
        self.assertIn("cumulative_excess_return", metrics)
        self.assertIn("period_excess_hit_rate", metrics)
        self.assertAlmostEqual(metrics["period_excess_hit_rate"], 0.75, places=12)
        self.assertFalse(math.isnan(metrics["cumulative_excess_return"]))


class ExcessReturnConsistencyTests(unittest.TestCase):
    """
    Synthetic-data tests that validate cumulative, monthly, and annualized
    excess return definitions against manually computed expected values.
    """

    def test_cumulative_excess_equals_cumulative_fund_minus_cumulative_benchmark(self) -> None:
        fund = pd.Series([0.01, -0.02, 0.03])
        benchmark = pd.Series([0.00, -0.01, 0.01])

        cum_fund, cum_bench, cum_excess = calculate_cumulative_excess_returns(fund, benchmark)

        # Manual calculation
        expected_cum_fund = pd.Series([
            (1.01) - 1,
            (1.01 * 0.98) - 1,
            (1.01 * 0.98 * 1.03) - 1,
        ])
        expected_cum_bench = pd.Series([
            (1.00) - 1,
            (1.00 * 0.99) - 1,
            (1.00 * 0.99 * 1.01) - 1,
        ])
        expected_cum_excess = expected_cum_fund - expected_cum_bench

        np.testing.assert_allclose(cum_fund.values, expected_cum_fund.values, atol=1e-12)
        np.testing.assert_allclose(cum_bench.values, expected_cum_bench.values, atol=1e-12)
        np.testing.assert_allclose(cum_excess.values, expected_cum_excess.values, atol=1e-12)

        # Verify it is NOT the same as compounding period excess
        period_excess = fund - benchmark
        compounded = (1 + period_excess).cumprod() - 1
        self.assertFalse(
            np.allclose(cum_excess.values, compounded.values, atol=1e-6),
            "Cumulative excess must NOT equal compounded period excess.",
        )

    def test_monthly_excess_equals_monthly_fund_minus_monthly_benchmark(self) -> None:
        fund = pd.Series([0.01, -0.02, 0.03])
        benchmark = pd.Series([0.00, -0.01, 0.01])

        period_excess = calculate_period_excess_returns(fund, benchmark)
        expected = fund - benchmark

        np.testing.assert_allclose(period_excess.values, expected.values, atol=1e-12)

        # Verify each value individually
        self.assertAlmostEqual(period_excess.iloc[0], 0.01 - 0.00, places=14)
        self.assertAlmostEqual(period_excess.iloc[1], -0.02 - (-0.01), places=14)
        self.assertAlmostEqual(period_excess.iloc[2], 0.03 - 0.01, places=14)

    def test_annualized_excess_equals_annualized_fund_minus_annualized_benchmark(self) -> None:
        fund = pd.Series([0.01, -0.02, 0.03])
        benchmark = pd.Series([0.00, -0.01, 0.01])
        periods_per_year = 12.0

        ann_excess = calculate_annualized_excess_return(fund, benchmark, periods_per_year)

        # Manual calculation
        cum_growth_fund = (1.01) * (0.98) * (1.03)
        cum_growth_bench = (1.00) * (0.99) * (1.01)
        expected_ann_fund = cum_growth_fund ** (12.0 / 3) - 1
        expected_ann_bench = cum_growth_bench ** (12.0 / 3) - 1
        expected_ann_excess = expected_ann_fund - expected_ann_bench

        self.assertAlmostEqual(ann_excess, expected_ann_excess, places=12)

    def test_cumulative_excess_with_identical_returns_is_zero(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, 0.005])
        cum_fund, cum_bench, cum_excess = calculate_cumulative_excess_returns(returns, returns)
        np.testing.assert_allclose(cum_excess.values, 0.0, atol=1e-14)

    def test_cumulative_excess_with_zero_benchmark(self) -> None:
        fund = pd.Series([0.01, -0.02, 0.03])
        benchmark = pd.Series([0.00, 0.00, 0.00])

        cum_fund, cum_bench, cum_excess = calculate_cumulative_excess_returns(fund, benchmark)
        np.testing.assert_allclose(cum_excess.values, cum_fund.values, atol=1e-14)

    def test_period_table_excess_consistency_with_manual_computation(self) -> None:
        """Verify build_relative_period_table excess = fund_period - bench_period."""
        fund_levels = [100.0, 101.0, 98.98, 101.9494]
        bench_levels = [100.0, 100.0, 99.0, 99.99]
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-29", "2024-03-31"]),
            "nav": fund_levels,
            "benchmark": bench_levels,
        })

        monthly = build_relative_period_table(
            df=df, freq="M",
            fund_level_col="nav", benchmark_level_col="benchmark",
            fund_output_col="fund_monthly_return",
            benchmark_output_col="benchmark_monthly_return",
            excess_output_col="monthly_excess_return",
        )

        for idx in monthly.index:
            fund_val = monthly.loc[idx, "fund_monthly_return"]
            bench_val = monthly.loc[idx, "benchmark_monthly_return"]
            excess_val = monthly.loc[idx, "monthly_excess_return"]
            if pd.notna(fund_val) and pd.notna(bench_val) and pd.notna(excess_val):
                self.assertAlmostEqual(excess_val, fund_val - bench_val, places=12)


if __name__ == "__main__":
    unittest.main()
