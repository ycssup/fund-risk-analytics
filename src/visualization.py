import matplotlib.pyplot as plt

def plot_nav_and_drawdown(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # NAV 曲线
    axes[0].plot(df.index, df["nav"], label="NAV")
    axes[0].set_title("NAV Curve")
    axes[0].set_ylabel("NAV")
    axes[0].legend()
    axes[0].grid(True)

    # Drawdown 曲线
    axes[1].plot(df.index, df["drawdown"], label="Drawdown")
    axes[1].set_title("Drawdown Curve")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("output/charts/nav_drawdown.png")
    plt.close()

    
def plot_rolling_metrics(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df.index, df["rolling_vol"], label="Rolling Volatility")
    axes[0].set_title("Rolling Volatility")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df.index, df["rolling_sharpe"], label="Rolling Sharpe")
    axes[1].set_title("Rolling Sharpe")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("output/charts/rolling_metrics.png")
    plt.close()