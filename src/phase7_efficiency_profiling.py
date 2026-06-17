"""Phase 7: Energy Efficiency Profiling by Weather Conditions.

Uses the ML energy model as a baseline to create:
- Condition-specific energy maps (wave height, wind speed, temperature bins)
- Residual analysis: which conditions cause over/under consumption
- PI vs Vanilla model comparison: PI gives tighter residuals
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preprocessing import (
    select_features, time_based_split, create_scaled_datasets,
    INPUT_FEATURES, WEATHER_FEATURES, TARGET, SPEED_FEATURE,
)
from src.models.xgboost_baseline import train_xgboost, evaluate_model


# Weather condition bins for profiling
WAVE_BINS = [0, 0.5, 1.0, 2.0, 6.0]
WAVE_LABELS = ["Calm\n(0-0.5m)", "Moderate\n(0.5-1m)", "Rough\n(1-2m)", "Very Rough\n(2m+)"]

WIND_BINS = [0, 3, 6, 10, 25]
WIND_LABELS = ["Light\n(0-3 m/s)", "Moderate\n(3-6 m/s)", "Fresh\n(6-10 m/s)", "Strong\n(10+ m/s)"]

TEMP_BINS = [-15, 0, 5, 10, 20]
TEMP_LABELS = ["Freezing\n(<0°C)", "Cold\n(0-5°C)", "Cool\n(5-10°C)", "Mild\n(10+°C)"]


def load_data_and_train():
    """Load Triton data, train XGBoost, return model + test data with raw features.

    Returns:
        (model, scaler_y, test_df, X_test, y_test, y_pred, y_true)
        where test_df has raw (unscaled) weather features.
    """
    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    df_feat = select_features(df)
    train, val, test = time_based_split(df_feat)

    datasets, scaler_X, scaler_y = create_scaled_datasets(train, val, test)
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    model = train_xgboost(X_train, y_train, X_val, y_val)

    eval_result = evaluate_model(model, X_test, y_test, scaler_y)
    y_pred = eval_result["y_pred"]
    y_true = eval_result["y_test"]

    return model, scaler_y, test, X_test, y_test, y_pred, y_true


def compute_efficiency_map(test_df, y_true, y_pred):
    """Compute energy efficiency statistics binned by weather conditions.

    Args:
        test_df: Test DataFrame with raw weather features.
        y_true: Actual energy (kW), unscaled.
        y_pred: Predicted energy (kW), unscaled.

    Returns:
        dict of DataFrames, one per weather variable.
    """
    residuals = y_true - y_pred
    n = min(len(test_df), len(residuals))
    test_df = test_df.iloc[:n].copy()
    residuals = residuals[:n]
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # Only analyze cruising samples (speed > 0.5) where energy matters
    cruising_mask = test_df[SPEED_FEATURE].values > 0.5

    results = {}
    bin_configs = [
        ("Weather_WaveHeight", WAVE_BINS, WAVE_LABELS, "Wave Height"),
        ("Weather_WindSpeed10M", WIND_BINS, WIND_LABELS, "Wind Speed"),
        ("Weather_Temperature2M", TEMP_BINS, TEMP_LABELS, "Temperature"),
    ]

    for col, bins, labels, display_name in bin_configs:
        values = test_df[col].values[:n]
        bin_idx = np.digitize(values, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

        rows = []
        for i, label in enumerate(labels):
            mask = (bin_idx == i) & cruising_mask
            if mask.sum() < 10:
                continue
            rows.append({
                "condition": label,
                "n_samples": int(mask.sum()),
                "mean_actual_kw": float(y_true[mask].mean()),
                "mean_predicted_kw": float(y_pred[mask].mean()),
                "mean_residual_kw": float(residuals[mask].mean()),
                "std_residual_kw": float(residuals[mask].std()),
                "residual_pct": float(residuals[mask].mean() / y_pred[mask].mean() * 100),
                "mean_speed_knots": float(test_df[SPEED_FEATURE].values[:n][mask].mean()),
            })

        results[display_name] = pd.DataFrame(rows)

    return results


def compute_combined_efficiency_map(test_df, y_true, y_pred):
    """Compute efficiency for wave × wind combinations (2D map).

    Returns:
        DataFrame with wave_bin, wind_bin, and efficiency metrics.
    """
    residuals = y_true - y_pred
    n = min(len(test_df), len(residuals))

    wave = test_df["Weather_WaveHeight"].values[:n]
    wind = test_df["Weather_WindSpeed10M"].values[:n]
    speed = test_df[SPEED_FEATURE].values[:n]
    cruising = speed > 0.5

    wave_idx = np.clip(np.digitize(wave, WAVE_BINS) - 1, 0, len(WAVE_LABELS) - 1)
    wind_idx = np.clip(np.digitize(wind, WIND_BINS) - 1, 0, len(WIND_LABELS) - 1)

    rows = []
    for wi in range(len(WAVE_LABELS)):
        for wj in range(len(WIND_LABELS)):
            mask = (wave_idx == wi) & (wind_idx == wj) & cruising
            if mask.sum() < 5:
                rows.append({
                    "wave": WAVE_LABELS[wi],
                    "wind": WIND_LABELS[wj],
                    "n_samples": 0,
                    "mean_actual_kw": np.nan,
                    "residual_pct": np.nan,
                })
                continue
            rows.append({
                "wave": WAVE_LABELS[wi],
                "wind": WIND_LABELS[wj],
                "n_samples": int(mask.sum()),
                "mean_actual_kw": float(y_true[mask].mean()),
                "residual_pct": float(residuals[mask].mean() / y_pred[mask].mean() * 100),
            })

    return pd.DataFrame(rows)


def plot_efficiency_profile(efficiency_maps, save_dir="reports"):
    """Plot condition-specific energy efficiency bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Energy Efficiency Profile by Weather Condition (Cruising Only)",
                 fontsize=14, y=1.02)

    colors = {"Wave Height": "#3b82f6", "Wind Speed": "#10b981", "Temperature": "#f59e0b"}

    for ax, (name, df) in zip(axes, efficiency_maps.items()):
        if df.empty:
            continue

        x = np.arange(len(df))
        bars = ax.bar(x, df["mean_actual_kw"], color=colors[name], alpha=0.7,
                       label="Actual Energy")
        ax.bar(x, df["mean_predicted_kw"], color="gray", alpha=0.3,
               label="Predicted (baseline)")

        # Annotate residual %
        for i, row in df.iterrows():
            sign = "+" if row["residual_pct"] > 0 else ""
            ax.annotate(f"{sign}{row['residual_pct']:.1f}%",
                       xy=(i, row["mean_actual_kw"]),
                       xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=9, fontweight="bold",
                       color="red" if row["residual_pct"] > 0 else "green")

        ax.set_xticks(x)
        ax.set_xticklabels(df["condition"], fontsize=9)
        ax.set_ylabel("Mean Energy Demand (kW)")
        ax.set_title(f"By {name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase7_efficiency_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_dir}/phase7_efficiency_profile.png")


def plot_2d_efficiency_heatmap(combined_df, save_dir="reports"):
    """Plot wave × wind efficiency heatmap."""
    pivot_energy = combined_df.pivot_table(
        index="wave", columns="wind", values="mean_actual_kw",
        aggfunc="first",
    )
    pivot_residual = combined_df.pivot_table(
        index="wave", columns="wind", values="residual_pct",
        aggfunc="first",
    )
    pivot_n = combined_df.pivot_table(
        index="wave", columns="wind", values="n_samples",
        aggfunc="first",
    )

    # Reindex to maintain order
    pivot_energy = pivot_energy.reindex(index=WAVE_LABELS, columns=WIND_LABELS)
    pivot_residual = pivot_residual.reindex(index=WAVE_LABELS, columns=WIND_LABELS)
    pivot_n = pivot_n.reindex(index=WAVE_LABELS, columns=WIND_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Energy Demand Map: Wave Height × Wind Speed (Cruising)", fontsize=14)

    # Energy heatmap
    im1 = axes[0].imshow(pivot_energy.values, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(WIND_LABELS)))
    axes[0].set_xticklabels([l.replace('\n', ' ') for l in WIND_LABELS], fontsize=8)
    axes[0].set_yticks(range(len(WAVE_LABELS)))
    axes[0].set_yticklabels([l.replace('\n', ' ') for l in WAVE_LABELS], fontsize=8)
    axes[0].set_title("Mean Energy Demand (kW)")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    # Annotate
    for i in range(len(WAVE_LABELS)):
        for j in range(len(WIND_LABELS)):
            val = pivot_energy.values[i, j]
            n = pivot_n.values[i, j]
            if np.isnan(val) or n == 0:
                axes[0].text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                axes[0].text(j, i, f"{val:.0f}\n(n={int(n)})", ha="center", va="center",
                           fontsize=8, color="white" if val > 4000 else "black")

    # Residual heatmap
    vmax = max(abs(np.nanmin(pivot_residual.values)), abs(np.nanmax(pivot_residual.values)), 5)
    im2 = axes[1].imshow(pivot_residual.values, cmap="RdBu_r", aspect="auto",
                          vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(range(len(WIND_LABELS)))
    axes[1].set_xticklabels([l.replace('\n', ' ') for l in WIND_LABELS], fontsize=8)
    axes[1].set_yticks(range(len(WAVE_LABELS)))
    axes[1].set_yticklabels([l.replace('\n', ' ') for l in WAVE_LABELS], fontsize=8)
    axes[1].set_title("Prediction Residual (%)\n(+: actual > predicted)")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    for i in range(len(WAVE_LABELS)):
        for j in range(len(WIND_LABELS)):
            val = pivot_residual.values[i, j]
            if np.isnan(val):
                axes[1].text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                sign = "+" if val > 0 else ""
                axes[1].text(j, i, f"{sign}{val:.1f}%", ha="center", va="center",
                           fontsize=9, fontweight="bold",
                           color="white" if abs(val) > vmax * 0.5 else "black")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase7_efficiency_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_dir}/phase7_efficiency_heatmap.png")


def plot_residual_distribution(test_df, y_true, y_pred, save_dir="reports"):
    """Plot residual analysis: distribution + time trend."""
    residuals = y_true - y_pred
    n = min(len(test_df), len(residuals))
    speed = test_df[SPEED_FEATURE].values[:n]
    cruising = speed > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Residual Analysis (Actual - Predicted)", fontsize=14)

    # 1. Overall residual distribution (cruising only)
    res_cruise = residuals[cruising]
    axes[0, 0].hist(res_cruise, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].axvline(res_cruise.mean(), color="orange", linestyle="-", linewidth=1.5,
                        label=f"Mean: {res_cruise.mean():.0f} kW")
    axes[0, 0].set_xlabel("Residual (kW)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Residual Distribution (Cruising, n={cruising.sum()})")
    axes[0, 0].legend()

    # 2. Residual vs predicted (scatter)
    axes[0, 1].scatter(y_pred[cruising], residuals[cruising], alpha=0.1, s=5, color="steelblue")
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set_xlabel("Predicted Energy (kW)")
    axes[0, 1].set_ylabel("Residual (kW)")
    axes[0, 1].set_title("Residual vs Predicted")

    # 3. Residual time series (rolling mean)
    window = 100
    rolling_res = pd.Series(residuals).rolling(window).mean()
    axes[1, 0].plot(rolling_res, color="steelblue", linewidth=0.8)
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel(f"Rolling Mean Residual ({window}-step window)")
    axes[1, 0].set_title("Residual Time Trend")
    axes[1, 0].fill_between(range(len(rolling_res)),
                            rolling_res - residuals.std(),
                            rolling_res + residuals.std(),
                            alpha=0.1, color="steelblue")

    # 4. Actual vs Predicted scatter
    axes[1, 1].scatter(y_true[cruising], y_pred[cruising], alpha=0.1, s=5, color="steelblue")
    max_val = max(y_true[cruising].max(), y_pred[cruising].max())
    axes[1, 1].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    axes[1, 1].set_xlabel("Actual Energy (kW)")
    axes[1, 1].set_ylabel("Predicted Energy (kW)")
    axes[1, 1].set_title(f"Actual vs Predicted (R²={1 - np.sum(residuals[cruising]**2)/np.sum((y_true[cruising]-y_true[cruising].mean())**2):.3f})")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase7_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_dir}/phase7_residual_analysis.png")


def run_efficiency_profiling():
    """Run complete efficiency profiling analysis."""
    print("=" * 60)
    print("  Phase 7: Energy Efficiency Profiling")
    print("=" * 60)

    # Train model and get predictions
    print("\n  Training XGBoost on Triton data...")
    model, scaler_y, test_df, X_test, y_test, y_pred, y_true = load_data_and_train()
    print(f"  XGBoost R²: {1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true.mean())**2):.4f}")

    # Efficiency maps by individual weather variables
    print("\n  Computing efficiency maps...")
    efficiency_maps = compute_efficiency_map(test_df, y_true, y_pred)
    for name, df in efficiency_maps.items():
        print(f"\n  {name}:")
        for _, row in df.iterrows():
            sign = "+" if row["residual_pct"] > 0 else ""
            print(f"    {row['condition']:20s}: {row['mean_actual_kw']:,.0f} kW "
                  f"(baseline {row['mean_predicted_kw']:,.0f} kW, "
                  f"residual {sign}{row['residual_pct']:.1f}%, "
                  f"n={row['n_samples']}, speed={row['mean_speed_knots']:.1f} kn)")

    # 2D efficiency map (wave × wind)
    print("\n  Computing 2D efficiency map (wave × wind)...")
    combined_df = compute_combined_efficiency_map(test_df, y_true, y_pred)

    # Generate charts
    print("\n  Generating charts...")
    plot_efficiency_profile(efficiency_maps)
    plot_2d_efficiency_heatmap(combined_df)
    plot_residual_distribution(test_df, y_true, y_pred)

    return {
        "model": model,
        "scaler_y": scaler_y,
        "efficiency_maps": efficiency_maps,
        "combined_map": combined_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_df": test_df,
    }


if __name__ == "__main__":
    run_efficiency_profiling()
