import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_nav_data(file_path: str) -> pd.DataFrame:
    """
    Read real NAV data from Excel.
    Assumes:
    - first column = date
    - second column = nav
    """
    df = pd.read_excel(file_path)

    # keep first 2 columns only
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "nav"]

    # clean data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    df = df.dropna(subset=["date", "nav"])
    df = df.drop_duplicates(subset="date")
    df = df.sort_values("date").reset_index(drop=True)

    return df


def generate_synthetic_nav(
    original_nav: pd.Series,
    seed: int = 42,
    trend_strength: float = 0.006,
    cycle_strength: float = 0.015,
    vol_scale: float = 1.25,
) -> np.ndarray:
    """
    Generate a synthetic NAV path that is intentionally independent from
    the original NAV path.
    """
    rng = np.random.default_rng(seed)
    original_nav = pd.to_numeric(original_nav, errors="coerce").dropna()
    original_log_returns = np.log(original_nav).diff().dropna()

    n = len(original_nav)
    sigma = original_log_returns.std()
    if np.isnan(sigma) or sigma == 0:
        sigma = 0.01

    trend = np.linspace(0, trend_strength * (n - 1), n)
    cycle = cycle_strength * np.sin(np.linspace(0, 5 * np.pi, n) + rng.uniform(0, 2 * np.pi))
    random_component = np.cumsum(rng.normal(loc=0.0, scale=sigma * vol_scale, size=n))
    log_nav = trend + cycle + random_component
    log_nav = log_nav - log_nav[0]

    if log_nav[-1] <= 0:
        log_nav = log_nav + np.linspace(0, abs(log_nav[-1]) + trend_strength * n, n)

    for i in range(1, n):
        if log_nav[i] - log_nav[i - 1] < -0.06:
            log_nav[i] = log_nav[i - 1] - 0.06

    nav = np.exp(log_nav)
    nav = nav / nav[0] * original_nav.iloc[0]

    return nav


def create_sanitized_nav(
    df: pd.DataFrame,
    max_abs_nav_corr: float = 0.35,
    max_abs_return_corr: float = 0.10,
    min_total_return: float = 0.80,
    max_seed: int = 10000,
) -> pd.DataFrame:
    """
    Create a sanitized NAV series with low NAV and return correlation to
    the original path while preserving the original date column.
    """
    df = df.copy()

    df["daily_return"] = df["nav"].pct_change().fillna(0)

    best_nav = None
    best_score = np.inf
    best_nav_corr = np.nan
    best_return_corr = np.nan
    best_seed = None

    for seed in range(max_seed):
        sanitized_nav = generate_synthetic_nav(df["nav"], seed=seed)
        sanitized_return = pd.Series(sanitized_nav).pct_change().fillna(0)
        total_return = sanitized_nav[-1] / sanitized_nav[0] - 1

        nav_corr = df["nav"].corr(pd.Series(sanitized_nav))
        return_corr = df["daily_return"].corr(sanitized_return)
        trend_penalty = max(min_total_return - total_return, 0)
        score = max(abs(nav_corr), abs(return_corr)) + trend_penalty

        if score < best_score:
            best_nav = sanitized_nav
            best_score = score
            best_nav_corr = nav_corr
            best_return_corr = return_corr
            best_seed = seed

        if (
            total_return >= min_total_return
            and abs(nav_corr) <= max_abs_nav_corr
            and abs(return_corr) <= max_abs_return_corr
        ):
            break

    sanitized_returns = pd.Series(best_nav).pct_change().fillna(0).to_numpy()

    df["sanitized_return"] = sanitized_returns
    df["sanitized_nav"] = best_nav
    df.attrs["sanitization_seed"] = best_seed
    df.attrs["nav_correlation"] = best_nav_corr
    df.attrs["return_correlation"] = best_return_corr

    return df


def save_sanitized_data(df: pd.DataFrame, output_csv: str, output_excel: str = None) -> None:
    """
    Save sanitized NAV series locally.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    export_df = df[["date", "sanitized_nav"]].copy()
    export_df.columns = ["date", "nav"]

    export_df.to_csv(output_csv, index=False)

    if output_excel:
        os.makedirs(os.path.dirname(output_excel), exist_ok=True)
        export_df.to_excel(output_excel, index=False)


def plot_comparison(df: pd.DataFrame, output_path: str = "output/charts/sanitized_nav_comparison.png") -> None:
    """
    Compare original NAV and sanitized NAV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["nav"], label="Original NAV")
    plt.plot(df["date"], df["sanitized_nav"], label="Sanitized NAV")
    plt.title("Original vs Sanitized NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def check_similarity(df: pd.DataFrame) -> None:
    """
    Print correlation between original and sanitized NAV/returns.
    """
    nav_corr = df["nav"].corr(pd.Series(df["sanitized_nav"]))
    corr = df["daily_return"].corr(pd.Series(df["sanitized_return"]))
    print(f"NAV correlation: {nav_corr:.4f}")
    print(f"Return correlation: {corr:.4f}")


if __name__ == "__main__":
    print("Step 1: loading real NAV data...")
    file_path = "data/real_nav.xlsx"
    df = load_nav_data(file_path)
    print(f"Loaded {len(df)} rows.")

    print("Step 2: creating sanitized NAV series...")
    df_sanitized = create_sanitized_nav(df)

    print("Step 3: saving sanitized files...")
    save_sanitized_data(
        df_sanitized,
        output_csv="data/nav_sanitized_low_corr.csv",
        output_excel="data/nav_sanitized_low_corr.xlsx"
    )

    print("Step 4: checking similarity...")
    check_similarity(df_sanitized)

    print("Step 5: plotting comparison...")
    plot_comparison(df_sanitized)

    print("Done. Sanitized NAV data saved successfully.")
