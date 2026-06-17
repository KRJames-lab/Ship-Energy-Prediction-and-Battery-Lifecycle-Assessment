"""Phase 4-5-6 Runner: SOC Simulation, Battery Degradation, SHAP Analysis.

Includes Charging Strategy Comparison (48 scenarios):
- 4 battery capacities × 3 strategies × 2 chemistries × 2 climates
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preprocessing import select_features, time_based_split, create_scaled_datasets, INPUT_FEATURES
from src.phase4_soc_simulation import (
    simulate_soc, compute_soc_statistics, BATTERY_SPECS,
    identify_voyage_segments, compute_voyage_energy_actual,
    build_adaptive_target_map, build_route_profile_data,
    CAPACITY_SWEEP, CHARGING_STRATEGIES, ROUTE_PROFILES,
)
from src.phase5_battery_degradation import simulate_degradation, simulate_degradation_stepwise
from src.phase6_shap_analysis import compute_shap_values, create_shap_charts, scenario_shap_comparison

TRITON_PERIOD_DAYS = 88  # Triton data spans ~88 days
TRITON_TEMP = 3.0        # Cold climate average (°C)
TROPICAL_TEMP = 25.0     # Poseidon tropical average (°C)


def run_phase4():
    """Phase 4: SOC Simulation (original, fixed_full only)."""
    print("=" * 60)
    print("  Phase 4: Battery SOC Simulation")
    print("=" * 60)

    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    p_battery = df["P_battery_kW"].values
    speed = df["Ship_SpeedOverGround"].fillna(0).values

    results = {}
    for chem in ["NMC", "LFP"]:
        print(f"\n  Simulating {chem}...")
        soc_result = simulate_soc(p_battery, speed, chemistry=chem)
        stats = compute_soc_statistics(soc_result)
        results[chem] = {"soc_result": soc_result, "stats": stats}

        print(f"    Mean SOC: {stats['mean_soc']:.3f}")
        print(f"    Min SOC:  {stats['min_soc']:.3f}")
        print(f"    Equivalent cycles: {stats['equivalent_cycles']:.1f}")
        print(f"    Charging: {stats['charge_time_pct']:.1f}%, Discharging: {stats['discharge_time_pct']:.1f}%")

    # Plot SOC comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Phase 4: Battery SOC Simulation (Triton)", fontsize=14)

    for i, chem in enumerate(["NMC", "LFP"]):
        soc = results[chem]["soc_result"]["soc"]
        stats = results[chem]["stats"]

        # Time series
        axes[i, 0].plot(soc[:2000], color="steelblue", linewidth=0.5)
        axes[i, 0].axhline(BATTERY_SPECS[chem]["soc_min"], color="red", linestyle="--", alpha=0.5, label=f"SOC min ({BATTERY_SPECS[chem]['soc_min']})")
        axes[i, 0].axhline(BATTERY_SPECS[chem]["soc_max"], color="green", linestyle="--", alpha=0.5, label=f"SOC max ({BATTERY_SPECS[chem]['soc_max']})")
        axes[i, 0].set_title(f"{chem}: SOC Time Series (first 2000 steps)")
        axes[i, 0].set_ylabel("SOC")
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].legend(fontsize=8)

        # Distribution
        axes[i, 1].hist(soc, bins=50, color="steelblue", edgecolor="white")
        axes[i, 1].set_title(f"{chem}: SOC Distribution (mean={stats['mean_soc']:.3f})")
        axes[i, 1].set_xlabel("SOC")

    plt.tight_layout()
    plt.savefig("reports/phase4_soc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Chart saved: reports/phase4_soc.png")

    return results


def run_charging_strategy_comparison():
    """Charging Strategy Comparison: 4 capacities × 3 strategies × 2 chemistries."""
    print("\n" + "=" * 60)
    print("  Phase 4+: Charging Strategy Comparison")
    print("=" * 60)

    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    p_battery = df["P_battery_kW"].values
    speed = df["Ship_SpeedOverGround"].fillna(0).values

    # Identify voyage segments for adaptive strategy
    segments = identify_voyage_segments(speed)
    segment_energies = compute_voyage_energy_actual(p_battery, segments)

    print(f"\n  Voyage segments: {len(segments)}")
    print(f"  Mean voyage energy: {segment_energies.mean():.0f} kWh")
    print(f"  Max voyage energy:  {segment_energies.max():.0f} kWh")

    # Run all scenarios
    all_results = {}
    strategies = ["fixed_full", "conservative", "weather_adaptive"]
    chemistries = ["NMC", "LFP"]

    print(f"\n  Running {len(CAPACITY_SWEEP)} capacities × {len(strategies)} strategies × {len(chemistries)} chemistries = {len(CAPACITY_SWEEP) * len(strategies) * len(chemistries)} SOC scenarios...")

    for cap in CAPACITY_SWEEP:
        for chem in chemistries:
            spec = BATTERY_SPECS[chem]
            for strat in strategies:
                # Build adaptive target map if needed
                adaptive_target = None
                if strat == "weather_adaptive":
                    adaptive_target = build_adaptive_target_map(
                        speed, p_battery, segments, segment_energies,
                        capacity_kwh=cap,
                        soc_min=spec["soc_min"],
                        soc_max=spec["soc_max"],
                        safety_margin=0.15,
                    )

                soc_result = simulate_soc(
                    p_battery, speed,
                    chemistry=chem,
                    strategy=strat,
                    capacity_kwh=cap,
                    adaptive_target_soc=adaptive_target,
                )
                stats = compute_soc_statistics(soc_result)

                key = f"{cap}kWh_{chem}_{strat}"
                all_results[key] = {"soc_result": soc_result, "stats": stats}

                print(f"    {key}: mean_SOC={stats['mean_soc']:.3f}, "
                      f"cycles={stats['equivalent_cycles']:.1f}, "
                      f"high_SOC={stats['time_at_high_soc_pct']:.1f}%, "
                      f"low_SOC={stats['time_at_low_soc_pct']:.1f}%, "
                      f"sweet={stats['time_in_sweet_spot_pct']:.1f}%")

    # Generate comparison charts
    _plot_strategy_comparison(all_results)
    _plot_capacity_sweep(all_results)

    return all_results


def run_degradation_with_strategies(soc_results: dict):
    """Phase 5+: Degradation with SOC/DoD stress for all scenarios."""
    print("\n" + "=" * 60)
    print("  Phase 5+: SOC-Dependent Degradation (48 scenarios)")
    print("=" * 60)

    degradation_results = {}
    temps = [(TRITON_TEMP, "Cold_3C"), (TROPICAL_TEMP, "Tropical_25C")]

    for soc_key, soc_data in soc_results.items():
        stats = soc_data["stats"]
        soc_result = soc_data["soc_result"]
        chem = stats["chemistry"]

        for temp, temp_label in temps:
            deg_key = f"{soc_key}_{temp_label}"
            result = simulate_degradation_stepwise(
                soc_timeseries=soc_result["soc"],
                cycle_dods=soc_result["cycle_dods"],
                period_days=TRITON_PERIOD_DAYS,
                temp_celsius=temp,
                chemistry=chem,
                years=10,
            )
            degradation_results[deg_key] = result

            eol = result["eol_year_80pct"]
            eol_str = f"{eol:.1f}y" if eol else ">10y"
            soh_10y = result["soh"][-1]
            print(f"    {deg_key}: SOH@10y={soh_10y:.1%}, EOL={eol_str}, "
                  f"k_soc={result['k_soc_stress']:.2f}, k_dod={result['k_dod_stress']:.2f}")

    # Generate degradation comparison charts
    _plot_degradation_strategies(degradation_results)
    _plot_degradation_heatmap(degradation_results)

    return degradation_results


def _plot_strategy_comparison(all_results: dict):
    """Plot SOC time series comparison for each capacity (3 strategies overlaid)."""
    fig, axes = plt.subplots(len(CAPACITY_SWEEP), 2, figsize=(18, 5 * len(CAPACITY_SWEEP)))
    fig.suptitle("Charging Strategy Comparison: SOC Time Series (NMC, first 3000 steps)", fontsize=14, y=1.01)

    strategy_colors = {"fixed_full": "steelblue", "conservative": "coral", "weather_adaptive": "forestgreen"}
    strategy_labels = {"fixed_full": "Fixed Full", "conservative": "Conservative (70%)", "weather_adaptive": "Weather-Adaptive"}

    for row, cap in enumerate(CAPACITY_SWEEP):
        for col, chem in enumerate(["NMC", "LFP"]):
            ax = axes[row, col]
            for strat in ["fixed_full", "conservative", "weather_adaptive"]:
                key = f"{cap}kWh_{chem}_{strat}"
                soc = all_results[key]["soc_result"]["soc"]
                ax.plot(soc[:3000], color=strategy_colors[strat], linewidth=0.7,
                        alpha=0.8, label=strategy_labels[strat])

            ax.set_title(f"{chem} — {cap:,} kWh", fontsize=10)
            ax.set_ylabel("SOC")
            ax.set_ylim(0, 1)
            ax.axhline(0.80, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
            ax.axhline(0.20, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
            ax.fill_between(range(3000), 0.20, 0.80, alpha=0.05, color="green")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("reports/phase4_soc_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Chart saved: reports/phase4_soc_strategies.png")


def _plot_capacity_sweep(all_results: dict):
    """Plot key metrics vs battery capacity for each strategy."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Battery Capacity Sweep: Impact on SOC Metrics", fontsize=14)

    strategy_colors = {"fixed_full": "steelblue", "conservative": "coral", "weather_adaptive": "forestgreen"}
    strategy_labels = {"fixed_full": "Fixed Full", "conservative": "Conservative", "weather_adaptive": "Adaptive"}
    metrics = [
        ("mean_soc", "Mean SOC"),
        ("equivalent_cycles", "Equivalent Cycles (88d)"),
        ("time_at_high_soc_pct", "Time at High SOC >80% (%)"),
        ("time_at_low_soc_pct", "Time at Low SOC <20% (%)"),
        ("time_in_sweet_spot_pct", "Time in Sweet Spot 20-80% (%)"),
        ("mean_dod", "Mean DoD per Cycle"),
    ]

    for col_idx, chem in enumerate(["NMC", "LFP"]):
        for ax_idx, (metric, label) in enumerate(metrics):
            row = ax_idx // 3
            col = ax_idx % 3
            ax = axes[row, col]

            for strat in ["fixed_full", "conservative", "weather_adaptive"]:
                values = []
                for cap in CAPACITY_SWEEP:
                    key = f"{cap}kWh_{chem}_{strat}"
                    values.append(all_results[key]["stats"][metric])

                marker = "o" if chem == "NMC" else "s"
                linestyle = "-" if chem == "NMC" else "--"
                ax.plot(CAPACITY_SWEEP, values, color=strategy_colors[strat],
                        marker=marker, linestyle=linestyle, linewidth=1.5,
                        label=f"{strategy_labels[strat]} ({chem})")

            ax.set_xlabel("Battery Capacity (kWh)")
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig("reports/phase4_capacity_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: reports/phase4_capacity_sweep.png")


def _plot_degradation_strategies(degradation_results: dict):
    """Plot SOH curves for all strategies at 60,000 kWh (where differences are largest)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("10-Year Battery Degradation: Charging Strategy Comparison (60,000 kWh)", fontsize=14)

    strategy_colors = {"fixed_full": "steelblue", "conservative": "coral", "weather_adaptive": "forestgreen"}
    strategy_labels = {"fixed_full": "Fixed Full", "conservative": "Conservative", "weather_adaptive": "Adaptive"}
    strategy_linestyles = {"fixed_full": "-", "conservative": "--", "weather_adaptive": "-."}

    for col, (temp_label, temp_title) in enumerate([("Cold_3C", "Cold Climate (3°C)"), ("Tropical_25C", "Tropical (25°C)")]):
        ax = axes[col]
        for chem in ["NMC", "LFP"]:
            for strat in ["fixed_full", "conservative", "weather_adaptive"]:
                key = f"60000kWh_{chem}_{strat}_{temp_label}"
                if key not in degradation_results:
                    continue
                result = degradation_results[key]
                chem_marker = "solid" if chem == "NMC" else "dashed"
                label = f"{chem} — {strategy_labels[strat]}"
                lw = 2.0 if chem == "NMC" else 1.5
                ax.plot(result["time_years"], result["soh"] * 100,
                        color=strategy_colors[strat],
                        linestyle="-" if chem == "NMC" else "--",
                        linewidth=lw, label=label)

        ax.axhline(80, color="gray", linestyle=":", alpha=0.7, label="EOL (80%)")
        ax.set_xlabel("Years")
        ax.set_ylabel("SOH (%)")
        ax.set_title(temp_title)
        ax.legend(fontsize=8)
        ax.set_ylim(50, 102)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/phase5_degradation_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: reports/phase5_degradation_strategies.png")


def _plot_degradation_heatmap(degradation_results: dict):
    """Plot 48-scenario heatmap: SOH@10y for all combinations."""
    strategies = ["fixed_full", "conservative", "weather_adaptive"]
    chemistries = ["NMC", "LFP"]
    temps = [("Cold_3C", "3°C"), ("Tropical_25C", "25°C")]

    # Build matrix: rows = capacity × chemistry × climate, cols = strategy
    row_labels = []
    data = []

    for cap in CAPACITY_SWEEP:
        for chem in chemistries:
            for temp_label, temp_display in temps:
                row_labels.append(f"{cap//1000}k {chem} {temp_display}")
                row_data = []
                for strat in strategies:
                    key = f"{cap}kWh_{chem}_{strat}_{temp_label}"
                    if key in degradation_results:
                        soh = degradation_results[key]["soh"][-1] * 100
                    else:
                        soh = np.nan
                    row_data.append(soh)
                data.append(row_data)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(8, len(row_labels) * 0.5)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=60, vmax=100)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(["Fixed Full", "Conservative", "Adaptive"], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(strategies)):
            val = data[i, j]
            color = "white" if val < 75 else "black"
            eol_marker = " *" if val <= 80 else ""
            ax.text(j, i, f"{val:.1f}%{eol_marker}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if val <= 80 else "normal")

    ax.set_title("SOH at 10 Years (%) — All 48 Scenarios\n* = reached EOL (≤80%)", fontsize=12)
    fig.colorbar(im, ax=ax, label="SOH (%)", shrink=0.8)

    plt.tight_layout()
    plt.savefig("reports/phase5_strategy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: reports/phase5_strategy_heatmap.png")


def run_route_profile_experiment():
    """Route Profile Experiment: Same ship, different operation patterns.

    Simulates 3 route profiles × realistic capacities × 3 strategies × 2 chemistries.
    Shows that adaptive charging is effective on short routes but not long cruises.
    """
    print("\n" + "=" * 60)
    print("  Route Profile Experiment")
    print("  'What if this ship operated on shorter routes?'")
    print("=" * 60)

    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    p_battery_orig = df["P_battery_kW"].values
    speed_orig = df["Ship_SpeedOverGround"].fillna(0).values

    # Original segments for reference
    segments_orig = identify_voyage_segments(speed_orig)
    energies_orig = compute_voyage_energy_actual(p_battery_orig, segments_orig)

    strategies = ["fixed_full", "conservative", "weather_adaptive"]
    chemistries = ["NMC", "LFP"]
    temps = [(TRITON_TEMP, "Cold_3C"), (TROPICAL_TEMP, "Tropical_25C")]

    all_soc_results = {}
    all_deg_results = {}

    for profile_name, profile_config in ROUTE_PROFILES.items():
        max_energy = profile_config["max_voyage_energy"]
        capacities = profile_config["realistic_capacities"]

        print(f"\n  --- {profile_name}: {profile_config['description']} ---")

        # Build route-specific data
        if max_energy is None:
            # Use original data
            p_battery = p_battery_orig
            speed = speed_orig
            n_voyages = len(segments_orig)
            period_days = TRITON_PERIOD_DAYS
        else:
            p_battery, speed, n_voyages, period_days = build_route_profile_data(
                p_battery_orig, speed_orig,
                segments_orig, energies_orig,
                max_voyage_energy=max_energy,
            )

        # Segments for this profile
        segments = identify_voyage_segments(speed)
        seg_energies = compute_voyage_energy_actual(p_battery, segments)

        print(f"    Qualifying voyages: {n_voyages}")
        print(f"    Synthetic period: {period_days:.1f} days, {len(p_battery)} steps")
        print(f"    Mean voyage energy: {seg_energies.mean():.0f} kWh")

        for cap in capacities:
            for chem in chemistries:
                spec = BATTERY_SPECS[chem]
                for strat in strategies:
                    # Build adaptive target
                    adaptive_target = None
                    if strat == "weather_adaptive":
                        adaptive_target = build_adaptive_target_map(
                            speed, p_battery, segments, seg_energies,
                            capacity_kwh=cap,
                            soc_min=spec["soc_min"],
                            soc_max=spec["soc_max"],
                            safety_margin=0.15,
                        )

                    soc_result = simulate_soc(
                        p_battery, speed,
                        chemistry=chem, strategy=strat,
                        capacity_kwh=cap,
                        adaptive_target_soc=adaptive_target,
                    )
                    stats = compute_soc_statistics(soc_result)
                    soc_key = f"{profile_name}_{cap}kWh_{chem}_{strat}"
                    all_soc_results[soc_key] = {"soc_result": soc_result, "stats": stats}

                    # Degradation for each climate
                    for temp, temp_label in temps:
                        deg_result = simulate_degradation(
                            cycles_per_period=stats["equivalent_cycles"],
                            period_days=period_days,
                            temp_celsius=temp,
                            chemistry=chem,
                            years=10,
                            mean_soc=stats["mean_soc"],
                            mean_dod=stats["mean_dod"],
                        )
                        deg_key = f"{soc_key}_{temp_label}"
                        all_deg_results[deg_key] = deg_result

                    print(f"    {soc_key}: SOC={stats['mean_soc']:.3f}, "
                          f"hi={stats['time_at_high_soc_pct']:.1f}%, "
                          f"lo={stats['time_at_low_soc_pct']:.1f}%, "
                          f"sweet={stats['time_in_sweet_spot_pct']:.1f}%, "
                          f"DoD={stats['mean_dod']:.3f}")

    # Generate route profile comparison charts
    _plot_route_profile_comparison(all_soc_results, all_deg_results)

    return all_soc_results, all_deg_results


def _plot_route_profile_comparison(soc_results: dict, deg_results: dict):
    """Plot the main result: route profile × strategy degradation comparison."""
    strategy_colors = {"fixed_full": "steelblue", "conservative": "coral", "weather_adaptive": "forestgreen"}
    strategy_labels = {"fixed_full": "Fixed Full", "conservative": "Conservative (70%)", "weather_adaptive": "Weather-Adaptive"}

    profiles = list(ROUTE_PROFILES.keys())

    # Chart 1: SOH@10y bar chart per route profile (NMC, Tropical — worst case)
    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 6))
    fig.suptitle("10-Year SOH by Route Profile & Charging Strategy (NMC, Tropical 25°C)", fontsize=13)

    for ax_idx, profile_name in enumerate(profiles):
        ax = axes[ax_idx]
        config = ROUTE_PROFILES[profile_name]
        capacities = config["realistic_capacities"]

        x = np.arange(len(capacities))
        width = 0.25

        for s_idx, strat in enumerate(["fixed_full", "conservative", "weather_adaptive"]):
            soh_vals = []
            for cap in capacities:
                deg_key = f"{profile_name}_{cap}kWh_NMC_{strat}_Tropical_25C"
                if deg_key in deg_results:
                    soh_vals.append(deg_results[deg_key]["soh"][-1] * 100)
                else:
                    soh_vals.append(np.nan)

            ax.bar(x + s_idx * width, soh_vals, width,
                   label=strategy_labels[strat], color=strategy_colors[strat])

        ax.axhline(80, color="gray", linestyle=":", alpha=0.7)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{c//1000}k" for c in capacities])
        ax.set_xlabel("Battery Capacity (kWh)")
        ax.set_ylabel("SOH at 10 Years (%)")
        ax.set_title(f"{profile_name.replace('_', ' ').title()}\n{config['description']}")
        ax.set_ylim(70, 100)
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("reports/phase5_route_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Chart saved: reports/phase5_route_profiles.png")

    # Chart 2: Key finding — strategy benefit vs route length
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Charging Strategy Benefit by Route Profile", fontsize=14)

    for col, chem in enumerate(["NMC", "LFP"]):
        ax = axes[col]
        profile_labels = []
        adaptive_benefit_cold = []
        adaptive_benefit_trop = []
        conservative_benefit_cold = []
        conservative_benefit_trop = []

        for profile_name in profiles:
            config = ROUTE_PROFILES[profile_name]
            cap = config["realistic_capacities"][-1]  # largest realistic capacity

            fixed_cold_key = f"{profile_name}_{cap}kWh_{chem}_fixed_full_Cold_3C"
            adapt_cold_key = f"{profile_name}_{cap}kWh_{chem}_weather_adaptive_Cold_3C"
            cons_cold_key = f"{profile_name}_{cap}kWh_{chem}_conservative_Cold_3C"
            fixed_trop_key = f"{profile_name}_{cap}kWh_{chem}_fixed_full_Tropical_25C"
            adapt_trop_key = f"{profile_name}_{cap}kWh_{chem}_weather_adaptive_Tropical_25C"
            cons_trop_key = f"{profile_name}_{cap}kWh_{chem}_conservative_Tropical_25C"

            if all(k in deg_results for k in [fixed_cold_key, adapt_cold_key, cons_cold_key,
                                                fixed_trop_key, adapt_trop_key, cons_trop_key]):
                profile_labels.append(f"{profile_name.replace('_', ' ').title()}\n({cap//1000}k kWh)")

                f_c = deg_results[fixed_cold_key]["soh"][-1]
                a_c = deg_results[adapt_cold_key]["soh"][-1]
                c_c = deg_results[cons_cold_key]["soh"][-1]
                f_t = deg_results[fixed_trop_key]["soh"][-1]
                a_t = deg_results[adapt_trop_key]["soh"][-1]
                c_t = deg_results[cons_trop_key]["soh"][-1]

                adaptive_benefit_cold.append((a_c - f_c) * 100)
                adaptive_benefit_trop.append((a_t - f_t) * 100)
                conservative_benefit_cold.append((c_c - f_c) * 100)
                conservative_benefit_trop.append((c_t - f_t) * 100)

        x = np.arange(len(profile_labels))
        width = 0.2

        ax.bar(x - 1.5 * width, adaptive_benefit_cold, width, label="Adaptive (3°C)", color="forestgreen", alpha=0.7)
        ax.bar(x - 0.5 * width, adaptive_benefit_trop, width, label="Adaptive (25°C)", color="forestgreen")
        ax.bar(x + 0.5 * width, conservative_benefit_cold, width, label="Conservative (3°C)", color="coral", alpha=0.7)
        ax.bar(x + 1.5 * width, conservative_benefit_trop, width, label="Conservative (25°C)", color="coral")

        ax.set_xticks(x)
        ax.set_xticklabels(profile_labels, fontsize=9)
        ax.set_ylabel("SOH Improvement vs Fixed Full (%p)")
        ax.set_title(f"{chem}: Strategy Benefit by Route Profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("reports/phase5_strategy_benefit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: reports/phase5_strategy_benefit.png")

    # Chart 3: SOC time series comparison for short_ferry (most interesting)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Short Ferry Route: SOC Time Series by Charging Strategy (first 3000 steps)", fontsize=13)

    for col, chem in enumerate(["NMC", "LFP"]):
        ax = axes[col]
        cap = ROUTE_PROFILES["short_ferry"]["realistic_capacities"][-1]

        for strat in ["fixed_full", "conservative", "weather_adaptive"]:
            key = f"short_ferry_{cap}kWh_{chem}_{strat}"
            if key in soc_results:
                soc = soc_results[key]["soc_result"]["soc"]
                ax.plot(soc[:3000], color=strategy_colors[strat], linewidth=0.7,
                        alpha=0.8, label=strategy_labels[strat])

        ax.axhline(0.80, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
        ax.axhline(0.20, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
        ax.fill_between(range(3000), 0.20, 0.80, alpha=0.05, color="green", label="Sweet spot")
        ax.set_title(f"{chem} — {cap//1000}k kWh")
        ax.set_ylabel("SOC")
        ax.set_xlabel("Time step (5 min)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("reports/phase4_short_ferry_soc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: reports/phase4_short_ferry_soc.png")


def run_phase5(phase4_results):
    """Phase 5: Arrhenius Battery Degradation (original, backward compatible)."""
    print("\n" + "=" * 60)
    print("  Phase 5: Arrhenius Battery Degradation (10-year)")
    print("=" * 60)

    degradation_results = {}

    for chem in ["NMC", "LFP"]:
        cycles = phase4_results[chem]["stats"]["equivalent_cycles"]
        print(f"\n  {chem} — {cycles:.1f} cycles in {TRITON_PERIOD_DAYS} days")

        for temp, temp_name in [(TRITON_TEMP, "Cold (3°C)"), (TROPICAL_TEMP, "Tropical (25°C)")]:
            key = f"{chem}_{temp_name}"
            result = simulate_degradation(
                cycles_per_period=cycles,
                period_days=TRITON_PERIOD_DAYS,
                temp_celsius=temp,
                chemistry=chem,
                years=10,
            )
            degradation_results[key] = result

            eol = result["eol_year_80pct"]
            eol_str = f"{eol:.1f} years" if eol else "> 10 years"
            soh_10y = result["soh"][-1]
            print(f"    {temp_name}: SOH@10y={soh_10y:.3f}, EOL(80%)={eol_str}")

    # Plot degradation curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 5: 10-Year Battery Degradation (Arrhenius Model)", fontsize=14)

    colors = {"Cold (3°C)": "steelblue", "Tropical (25°C)": "coral"}
    linestyles = {"NMC": "-", "LFP": "--"}

    for key, result in degradation_results.items():
        chem = result["chemistry"]
        temp_name = "Cold (3°C)" if result["temp_celsius"] == TRITON_TEMP else "Tropical (25°C)"
        label = f"{chem} — {temp_name}"
        axes[0].plot(result["time_years"], result["soh"] * 100,
                    color=colors[temp_name], linestyle=linestyles[chem],
                    linewidth=2, label=label)

    axes[0].axhline(80, color="gray", linestyle=":", alpha=0.7, label="EOL (80%)")
    axes[0].set_xlabel("Years")
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_title("State of Health Over Time")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(50, 102)
    axes[0].grid(True, alpha=0.3)

    bar_data = []
    for key, result in degradation_results.items():
        chem = result["chemistry"]
        temp_name = "Cold (3°C)" if result["temp_celsius"] == TRITON_TEMP else "Tropical (25°C)"
        bar_data.append({
            "label": f"{chem}\n{temp_name}",
            "calendar": result["soh_calendar_loss"][-1] * 100,
            "cycle": result["soh_cycle_loss"][-1] * 100,
        })

    x = np.arange(len(bar_data))
    cal_vals = [d["calendar"] for d in bar_data]
    cyc_vals = [d["cycle"] for d in bar_data]
    labels = [d["label"] for d in bar_data]

    axes[1].bar(x, cal_vals, 0.6, label="Calendar Aging", color="steelblue")
    axes[1].bar(x, cyc_vals, 0.6, bottom=cal_vals, label="Cycle Aging", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("SOH Loss (%)")
    axes[1].set_title("Aging Breakdown at 10 Years")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("reports/phase5_degradation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Chart saved: reports/phase5_degradation.png")

    return degradation_results


def run_phase6():
    """Phase 6: SHAP Analysis."""
    print("\n" + "=" * 60)
    print("  Phase 6: SHAP Feature Importance Analysis")
    print("=" * 60)

    print("\n  Loading model and data...")
    xgb_model = joblib.load("checkpoints/poseidon_xgboost.pkl")

    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    df_feat = select_features(df)
    train, val, test = time_based_split(df_feat)
    datasets, scaler_X, scaler_y = create_scaled_datasets(train, val, test)
    X_test, y_test = datasets["test"]
    speed_test = test["Ship_SpeedOverGround"].values

    print("  Computing SHAP values...")
    shap_values = compute_shap_values(xgb_model, X_test, INPUT_FEATURES)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    print("\n  Top 10 Features by |SHAP|:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {INPUT_FEATURES[idx]:40s} {mean_abs_shap[idx]:.4f}")

    print("\n  Generating SHAP charts...")
    create_shap_charts(shap_values, save_dir="reports")

    print("  Computing scenario comparisons...")
    scenarios = scenario_shap_comparison(
        xgb_model, X_test, speed_test, INPUT_FEATURES, save_dir="reports"
    )

    return shap_values, scenarios


def main():
    print("Phase 4-5-6: SOC, Degradation, SHAP Analysis")
    print("  + Charging Strategy Comparison (48 scenarios)\n")

    # Phase 4 (original)
    phase4_results = run_phase4()

    # Phase 4+: Charging Strategy Comparison
    strategy_results = run_charging_strategy_comparison()

    # Phase 5 (original)
    phase5_results = run_phase5(phase4_results)

    # Phase 5+: Degradation with SOC/DoD stress
    degradation_strategy_results = run_degradation_with_strategies(strategy_results)

    # Phase 6: SHAP
    shap_values, scenarios = run_phase6()

    # Summary
    print("\n" + "=" * 60)
    print("  All Phases Complete!")
    print("=" * 60)
    print("\n  Generated reports:")
    print("    reports/phase4_soc.png                  — SOC simulation (original)")
    print("    reports/phase4_soc_strategies.png        — Strategy comparison SOC")
    print("    reports/phase4_capacity_sweep.png        — Capacity sweep metrics")
    print("    reports/phase5_degradation.png           — 10-year degradation (original)")
    print("    reports/phase5_degradation_strategies.png — Strategy degradation curves")
    print("    reports/phase5_strategy_heatmap.png      — 48-scenario heatmap")
    print("    reports/phase6_shap_bar.png              — SHAP bar chart")
    print("    reports/phase6_shap_beeswarm.png         — SHAP beeswarm")
    print("    reports/phase6_shap_scenarios.png        — Scenario comparison")


if __name__ == "__main__":
    main()
