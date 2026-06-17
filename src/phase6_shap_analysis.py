"""Phase 6: SHAP Analysis for model interpretation.

Applies SHAP to the best-performing XGBoost model to:
1. Rank weather variable importance
2. Compare feature importance across scenarios
"""
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preprocessing import INPUT_FEATURES


def compute_shap_values(model, X_test, feature_names=None, max_samples=2000):
    """Compute SHAP values for XGBoost model.

    Args:
        model: Trained XGBoost model.
        X_test: Test features array.
        feature_names: List of feature names.
        max_samples: Max samples for SHAP computation.

    Returns:
        shap.Explanation object.
    """
    if feature_names is None:
        feature_names = INPUT_FEATURES

    # Subsample if needed
    if len(X_test) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    shap_values.feature_names = feature_names

    return shap_values


def create_shap_charts(shap_values, save_dir="reports"):
    """Create SHAP summary and bar charts."""
    # 1. Bar chart (mean |SHAP|)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase6_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase6_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP charts saved to {save_dir}/")


def scenario_shap_comparison(model, X_test, speed, feature_names=None, save_dir="reports"):
    """Compare SHAP values across scenarios.

    Scenarios:
    - Cruising (speed >= 2 knots) vs Stationary (speed < 0.5 knots)
    - High wind vs Low wind
    """
    if feature_names is None:
        feature_names = INPUT_FEATURES

    explainer = shap.TreeExplainer(model)
    wind_idx = feature_names.index("Weather_WindSpeed10M")

    scenarios = {}

    # Cruising vs Stationary
    cruise_mask = speed >= 2.0
    station_mask = speed < 0.5

    for name, mask in [("Cruising", cruise_mask), ("Stationary", station_mask)]:
        X_sub = X_test[mask]
        if len(X_sub) > 1000:
            idx = np.random.RandomState(42).choice(len(X_sub), 1000, replace=False)
            X_sub = X_sub[idx]
        if len(X_sub) > 0:
            sv = explainer(X_sub)
            sv.feature_names = feature_names
            mean_abs = np.abs(sv.values).mean(axis=0)
            scenarios[name] = dict(zip(feature_names, mean_abs))

    # High wind vs Low wind
    wind_vals = X_test[:, wind_idx]
    wind_median = np.median(wind_vals)
    high_wind = X_test[wind_vals > wind_median]
    low_wind = X_test[wind_vals <= wind_median]

    for name, X_sub in [("High Wind", high_wind), ("Low Wind", low_wind)]:
        if len(X_sub) > 1000:
            idx = np.random.RandomState(42).choice(len(X_sub), 1000, replace=False)
            X_sub = X_sub[idx]
        if len(X_sub) > 0:
            sv = explainer(X_sub)
            sv.feature_names = feature_names
            mean_abs = np.abs(sv.values).mean(axis=0)
            scenarios[name] = dict(zip(feature_names, mean_abs))

    # Plot comparison
    if len(scenarios) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Cruising vs Stationary
        if "Cruising" in scenarios and "Stationary" in scenarios:
            top_features = sorted(
                scenarios["Cruising"].keys(),
                key=lambda f: scenarios["Cruising"][f], reverse=True
            )[:10]
            y_pos = np.arange(len(top_features))
            axes[0].barh(y_pos - 0.2, [scenarios["Cruising"][f] for f in top_features],
                        0.4, label="Cruising", color="steelblue")
            axes[0].barh(y_pos + 0.2, [scenarios["Stationary"].get(f, 0) for f in top_features],
                        0.4, label="Stationary", color="coral")
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(top_features, fontsize=8)
            axes[0].set_title("Cruising vs Stationary")
            axes[0].legend()
            axes[0].invert_yaxis()

        # High Wind vs Low Wind
        if "High Wind" in scenarios and "Low Wind" in scenarios:
            top_features = sorted(
                scenarios["High Wind"].keys(),
                key=lambda f: scenarios["High Wind"][f], reverse=True
            )[:10]
            y_pos = np.arange(len(top_features))
            axes[1].barh(y_pos - 0.2, [scenarios["High Wind"][f] for f in top_features],
                        0.4, label="High Wind", color="steelblue")
            axes[1].barh(y_pos + 0.2, [scenarios["Low Wind"].get(f, 0) for f in top_features],
                        0.4, label="Low Wind", color="coral")
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(top_features, fontsize=8)
            axes[1].set_title("High Wind vs Low Wind")
            axes[1].legend()
            axes[1].invert_yaxis()

        plt.suptitle("SHAP Feature Importance: Scenario Comparison", fontsize=14)
        plt.tight_layout()
        fig.savefig(f"{save_dir}/phase6_shap_scenarios.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Scenario comparison saved: {save_dir}/phase6_shap_scenarios.png")

    return scenarios
