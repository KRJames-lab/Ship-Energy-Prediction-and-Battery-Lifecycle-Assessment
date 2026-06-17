"""Phase 2 evaluation: compare models and generate charts."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred):
    """Compute regression metrics in original scale."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 100.0  # exclude near-zero values for MAPE
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def compute_physics_violations(y_pred_orig, speed_orig, wave_height_orig, p_hotel):
    """Compute physics constraint violation rates."""
    violations = {}

    # Hotel load violation: P < P_hotel
    hotel_mask = y_pred_orig < p_hotel
    violations["hotel_violation_pct"] = hotel_mask.mean() * 100

    # Cube law: check if power increases with speed (correlation)
    valid = speed_orig > 0.5  # exclude stationary
    if valid.sum() > 10:
        corr = np.corrcoef(speed_orig[valid] ** 3, y_pred_orig[valid])[0, 1]
        violations["cube_correlation"] = corr
    else:
        violations["cube_correlation"] = np.nan

    return violations


def create_comparison_chart(results, save_path="reports/phase2_comparison.png"):
    """Create comparison charts for all models.

    Args:
        results: dict of {model_name: {"metrics": {...}, "y_pred": array, "y_test": array}}
    """
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Phase 2: Poseidon Pre-training Model Comparison", fontsize=14, y=1.02)

    for i, (name, res) in enumerate(results.items()):
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        metrics = res["metrics"]

        # 1. Predicted vs Actual scatter
        axes[i, 0].scatter(y_test, y_pred, alpha=0.1, s=3, color="steelblue")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[i, 0].plot(lims, lims, "r--", linewidth=1)
        axes[i, 0].set_title(f"{name}: Predicted vs Actual")
        axes[i, 0].set_xlabel("Actual P_battery (kW)")
        axes[i, 0].set_ylabel("Predicted P_battery (kW)")
        axes[i, 0].text(
            0.05, 0.95,
            f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.0f}\nMAE={metrics['MAE']:.0f}",
            transform=axes[i, 0].transAxes, verticalalignment="top",
            fontsize=9, bbox={"boxstyle": "round", "alpha": 0.3},
        )

        # 2. Residual distribution
        residuals = y_pred - y_test
        axes[i, 1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
        axes[i, 1].axvline(0, color="red", linestyle="--")
        axes[i, 1].set_title(f"{name}: Residual Distribution")
        axes[i, 1].set_xlabel("Residual (kW)")

        # 3. Time series comparison (first 500 points)
        n_show = min(500, len(y_test))
        axes[i, 2].plot(y_test[:n_show], label="Actual", alpha=0.7, linewidth=0.8)
        axes[i, 2].plot(y_pred[:n_show], label="Predicted", alpha=0.7, linewidth=0.8)
        axes[i, 2].set_title(f"{name}: Time Series (first {n_show})")
        axes[i, 2].set_xlabel("Time Step")
        axes[i, 2].set_ylabel("P_battery (kW)")
        axes[i, 2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved: {save_path}")


def print_comparison_table(results):
    """Print formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE%':>10}")
    print(f"{'='*70}")
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<25} {m['RMSE']:>10.1f} {m['MAE']:>10.1f} {m['R2']:>10.4f} {m['MAPE']:>10.2f}")
    print(f"{'='*70}")
