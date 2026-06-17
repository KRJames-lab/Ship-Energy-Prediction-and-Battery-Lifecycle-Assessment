"""Phase 1 Runner: Apply energy conversion to both datasets and validate."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.phase1_energy_conversion import convert_triton, convert_poseidon

OUTPUT_DIR = "Dataset"
REPORT_DIR = "reports"


def validate_and_report(df: pd.DataFrame, name: str) -> dict:
    """Print validation statistics for converted dataset."""
    stats = {
        "name": name,
        "rows": len(df),
        "P_battery_kW_mean": df["P_battery_kW"].mean(),
        "P_battery_kW_std": df["P_battery_kW"].std(),
        "P_battery_kW_min": df["P_battery_kW"].min(),
        "P_battery_kW_max": df["P_battery_kW"].max(),
        "P_battery_kW_median": df["P_battery_kW"].median(),
        "negative_count": (df["P_battery_kW"] < 0).sum(),
        "nan_count": df["P_battery_kW"].isna().sum(),
        "zero_count": (df["P_battery_kW"] == 0).sum(),
        "total_energy_MWh": df["energy_kWh"].sum() / 1000,
    }

    print(f"\n{'='*60}")
    print(f"  {name} Energy Conversion Results")
    print(f"{'='*60}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>12.2f}")
        else:
            print(f"  {k:30s}: {v:>12}")
    print(f"{'='*60}")

    return stats


def create_validation_chart(triton_df, poseidon_df):
    """Create validation charts for both datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 1: Diesel → Electric Energy Conversion Validation", fontsize=14)

    for i, (df, name) in enumerate([(triton_df, "Triton"), (poseidon_df, "Poseidon")]):
        # Distribution
        axes[i, 0].hist(df["P_battery_kW"], bins=50, color="steelblue", edgecolor="white")
        axes[i, 0].set_title(f"{name}: P_battery Distribution")
        axes[i, 0].set_xlabel("P_battery (kW)")
        axes[i, 0].set_ylabel("Count")

        # Time series (sample)
        sample = df["P_battery_kW"].iloc[:500]
        axes[i, 1].plot(sample.values, color="steelblue", linewidth=0.5)
        axes[i, 1].set_title(f"{name}: P_battery Time Series (first 500)")
        axes[i, 1].set_xlabel("Time Step (5-min)")
        axes[i, 1].set_ylabel("P_battery (kW)")

        # Speed vs Power scatter (if speed column exists)
        speed_col = "Ship_SpeedOverGround"
        if speed_col in df.columns:
            sample_idx = np.random.RandomState(42).choice(len(df), min(2000, len(df)), replace=False)
            axes[i, 2].scatter(
                df[speed_col].iloc[sample_idx],
                df["P_battery_kW"].iloc[sample_idx],
                alpha=0.3, s=5, color="steelblue",
            )
            axes[i, 2].set_title(f"{name}: Speed vs Power")
            axes[i, 2].set_xlabel("Speed Over Ground (knots)")
            axes[i, 2].set_ylabel("P_battery (kW)")

    plt.tight_layout()
    path = f"{REPORT_DIR}/phase1_validation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nValidation chart saved: {path}")


def main():
    print("Loading datasets...")
    triton_raw = pd.read_parquet(f"{OUTPUT_DIR}/CPS_Triton.parquet")
    poseidon_raw = pd.read_parquet(f"{OUTPUT_DIR}/CPS_Poseidon.parquet")

    print("Converting Triton (HVO 30, single fuel)...")
    triton = convert_triton(triton_raw)
    triton_stats = validate_and_report(triton, "Triton (HVO 30)")

    print("Converting Poseidon (DM/RM, multi-fuel)...")
    poseidon = convert_poseidon(poseidon_raw)
    poseidon_stats = validate_and_report(poseidon, "Poseidon (Multi-fuel)")

    # Idle power estimation (hotel load) - speed < 0.5 knots
    print("\n--- Hotel Load Estimation (speed < 0.5 knots) ---")
    for df, name in [(triton, "Triton"), (poseidon, "Poseidon")]:
        if "Ship_SpeedOverGround" in df.columns:
            idle = df[df["Ship_SpeedOverGround"] < 0.5]["P_battery_kW"]
            if len(idle) > 0:
                print(f"  {name}: median={idle.median():.1f} kW, "
                      f"mean={idle.mean():.1f} kW, count={len(idle)}")

    # Save
    triton_path = f"{OUTPUT_DIR}/CPS_Triton_with_energy.parquet"
    poseidon_path = f"{OUTPUT_DIR}/CPS_Poseidon_with_energy.parquet"
    triton.to_parquet(triton_path, index=False)
    poseidon.to_parquet(poseidon_path, index=False)
    print(f"\nSaved: {triton_path}")
    print(f"Saved: {poseidon_path}")

    # Validation chart
    create_validation_chart(triton, poseidon)

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
