# Fund NAV-Based Risk Metrics Analysis

## Project Overview

This project focuses on calculating and analyzing key risk metrics for a fund product using time-series Net Asset Value (NAV) data, from the perspective of an investment risk manager.

The objective is to transform NAV data into actionable insights on return, volatility, drawdown, and downside risk.

---

## Key Questions Addressed

* How does the fund perform over time?
* What is the level of volatility and downside risk?
* How severe are historical drawdowns?
* Is the return justified relative to the risk taken?

---

## Methodology

1. Convert NAV series into periodic returns
2. Compute key risk metrics
3. Analyze drawdown dynamics
4. Evaluate risk-adjusted performance
5. Visualize results for interpretation

---

## Risk Metrics Covered

* Annualized Return
* Annualized Volatility
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Value at Risk (VaR, 95%)

---

## Project Structure

```bash
fund-NAV-based-risk-metrics-analysis/
├── data/
│   └── raw/
│       └── fund_nav.csv
├── notebooks/
│   └── 01_nav_risk_analysis.ipynb
├── src/
├── outputs/
└── README.md
```

---

## Tools & Technologies

* Python
* pandas
* numpy
* matplotlib
* Jupyter Notebook

---

## Key Insights (To Be Updated)

* To be added after completing the analysis

---

## Future Improvements

* Rolling risk metrics (rolling Sharpe, rolling volatility)
* Benchmark comparison
* Stress testing and scenario analysis
* Multi-fund comparison framework
* Liquidity risk analysis

---

## Author Perspective

This project reflects a structured approach to fund risk evaluation, aligning with real-world investment risk management practices rather than purely academic analysis.
