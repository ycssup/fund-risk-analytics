# Fund Risk Analytics & Relative Performance Monitoring System

Institutional-grade NAV analytics pipeline for benchmark-relative risk and performance monitoring, with automated visualization and PDF reporting.

## Project Overview

This system addresses a core buy-side requirement: evaluating fund performance and risk in absolute terms and relative to the investment benchmark on a consistent mathematical basis.

NAV-based analysis is the operational backbone of portfolio surveillance because it captures realized path behavior, not just terminal outcomes. Risk is path-dependent: two strategies can share a similar endpoint while exhibiting materially different drawdown depth, recovery profile, and volatility regime.

Benchmark-relative analysis is equally critical. Active portfolios are judged on excess return quality, not standalone return levels. The framework in this repository formalizes period, cumulative, and annualized excess return definitions to prevent category errors between point-in-time comparisons and compounded path outcomes.

## Key Features

- NAV-based return and risk analytics across multiple frequencies
- Drawdown and path-dependent risk analysis with recovery context
- Relative performance framework spanning period, cumulative, and annualized excess return
- Rolling risk and performance monitoring for regime diagnostics
- Visualization outputs including line charts, bar charts, and monthly heatmap tables
- Automated report generation to production-style PDF deliverables

## Relative Performance Framework

This repository uses strict, explicit definitions:

- Period Excess Return = Fund Period Return - Benchmark Period Return
- Cumulative Excess Return = Cumulative Fund Return - Cumulative Benchmark Return
- Annualized Excess Return = Annualized Fund Return - Annualized Benchmark Return

Mathematically:

- Cumulative Return: `(1 + return_series).cumprod() - 1`
- Cumulative Excess Return is computed as the difference of the two cumulative return series, not as the compounded product of period excess returns.

Why this distinction matters:

- Period excess is a point-in-time spread.
- Cumulative excess is path-dependent because compounding is nonlinear.
- Annualized excess is a horizon-normalized comparison derived from geometric annualization.

Conflating these definitions can distort active performance interpretation, especially in volatile return paths where compounding effects are material.

## Example Outputs

- Fund vs Benchmark Chart: Rebased level comparison highlights relative trend persistence, turning points, and divergence windows.
- Cumulative Excess Return Line: Isolates active value creation or erosion through time and shows whether excess performance is persistent, mean-reverting, or regime-dependent.
- Monthly Excess Return Heatmap: Exposes seasonality, concentration of alpha months, and dispersion across calendar periods, supporting hit-rate and consistency diagnostics.

## Project Structure

```text
data/                  # Fund NAV and benchmark input files
src/                   # Core analytics modules and pipeline orchestration
scripts/               # CLI entry points for analysis and report generation
output/                # Generated charts and report artifacts
output/reports/        # PDF and tabular reporting outputs
tests/                 # Unit tests for analytical correctness
```

Folder purpose summary:

- `data/`: Source time series used by the pipeline
- `src/`: Metric engines, alignment logic, narrative, and visualization components
- `scripts/`: Execution layer for production-style runs
- `output/`: Persisted deliverables for review and distribution
- `output/reports/`: Final reporting package, including PDF output

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run analytics pipeline:

```bash
python scripts/run_analysis.py --input data/sample_nav_data.xlsx --benchmark "data/benchmark_CSI 300.xlsx" --benchmark-name "CSI 300 Index" --output output
```

Generate PDF report:

```bash
python scripts/generate_report.py --input data/sample_nav_data.xlsx --benchmark "data/benchmark_CSI 300.xlsx" --benchmark-name "CSI 300 Index" --output output/reports/fund_risk_report.pdf
```

## Tech Stack

- Python
- pandas
- numpy
- matplotlib

## Future Enhancements

- Rolling excess return monitoring dashboards with regime segmentation
- Enhanced benchmark alignment alternatives and side-by-side sensitivity diagnostics
- LLM-based narrative reporting for executive summaries and risk commentary automation

## Positioning Statement

This repository is positioned as a buy-side style risk analytics and performance monitoring system for institutional NAV surveillance, benchmark-relative attribution, and reporting automation.
