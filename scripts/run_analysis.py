import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_nav_data
from src.return_calculator import calculate_daily_returns, calculate_cumulative_returns
from src.risk_metrics import annualized_return, annualized_volatility, sharpe_ratio
from src.drawdown import calculate_drawdown, max_drawdown
from src.visualization import plot_nav_and_drawdown
from src.rolling_metrics import rolling_volatility, rolling_sharpe
from src.visualization import plot_rolling_metrics

if __name__ == "__main__":
    file_path = "data/sample_nav.csv"

    # Step 1: load NAV data
    df = load_nav_data(file_path)

    # Step 2: calculate returns
    df = calculate_daily_returns(df)
    df = calculate_cumulative_returns(df)

    # Step 3: rolling metrics
    window = 20
    df["rolling_vol"] = rolling_volatility(df, window=window)
    df["rolling_sharpe"] = rolling_sharpe(df, window=window)

    # Step 4: calculate risk metrics
    ann_ret = annualized_return(df)
    ann_vol = annualized_volatility(df)
    sr = sharpe_ratio(df)

    # Step 5: calculate drawdown
    df = calculate_drawdown(df)
    md = max_drawdown(df)

    # Step 6: visualization
    plot_nav_and_drawdown(df)
    plot_rolling_metrics(df)

    print("=== NAV Data with Returns ===")
    print(df.round(6))
    print("\n=== Risk Metrics ===")
    print(f"Annualized Return: {ann_ret:.4%}")
    print(f"Annualized Volatility: {ann_vol:.4%}")
    print(f"Sharpe Ratio: {sr:.4f}")
    print(f"Maximum Drawdown: {md:.4%}")