"""Phase 4: Battery SOC (State of Charge) Simulation with Charging Strategies.

Simulates battery charge/discharge cycles using Triton energy demand:
- Cruising (speed > 0.5 knots) → discharge from battery
- Stationary (speed <= 0.5 knots) → charge from shore power

Charging Strategies:
- fixed_full: Always charge to SOC_max (baseline)
- conservative: Charge to 70% only (longevity priority)
- weather_adaptive: ML-predicted demand + safety margin (smart charging)

Battery chemistries: NMC (Nickel-Manganese-Cobalt), LFP (Lithium Iron Phosphate)
"""
import numpy as np
import pandas as pd

# Battery specifications
BATTERY_SPECS = {
    "NMC": {
        "capacity_kWh": 5000,       # Total battery capacity
        "soc_min": 0.15,            # Minimum SOC (15%)
        "soc_max": 0.95,            # Maximum SOC (95%)
        "charge_rate_kW": 2000,     # Shore charging power
        "discharge_efficiency": 0.95,  # Battery discharge efficiency
        "charge_efficiency": 0.92,     # Charging efficiency
        "nominal_voltage": 800,     # V
        "cycles_to_80pct": 2000,    # Cycle life to 80% SOH
    },
    "LFP": {
        "capacity_kWh": 5000,
        "soc_min": 0.10,            # LFP tolerates deeper discharge
        "soc_max": 0.98,
        "charge_rate_kW": 2000,
        "discharge_efficiency": 0.96,  # Slightly better than NMC
        "charge_efficiency": 0.93,
        "nominal_voltage": 800,
        "cycles_to_80pct": 5000,    # Much longer cycle life
    },
}

INTERVAL_HOURS = 5 / 60  # 5-minute intervals

# Charging strategy definitions
CHARGING_STRATEGIES = {
    "fixed_full": {"description": "Always charge to SOC_max"},
    "conservative": {"description": "Charge to 70% SOC only", "target_soc": 0.70},
    "weather_adaptive": {"description": "ML-predicted demand + safety margin"},
}

CAPACITY_SWEEP = [5_000, 15_000, 30_000, 60_000]  # kWh options

# Route profiles: filter voyages by max energy to simulate different operation patterns
ROUTE_PROFILES = {
    "long_cruise": {
        "description": "Original Triton pattern (all voyages)",
        "max_voyage_energy": None,  # no filter
        "realistic_capacities": [15_000, 30_000],  # kWh
    },
    "short_ferry": {
        "description": "Short-haul ferry (<5,000 kWh per voyage)",
        "max_voyage_energy": 5_000,
        "realistic_capacities": [5_000, 10_000, 15_000],
    },
    "coastal_mixed": {
        "description": "Coastal routes (<10,000 kWh per voyage)",
        "max_voyage_energy": 10_000,
        "realistic_capacities": [10_000, 15_000, 20_000],
    },
}


def build_route_profile_data(
    p_battery_kw: np.ndarray,
    speed: np.ndarray,
    segments: list,
    segment_energies: np.ndarray,
    max_voyage_energy: float,
    min_total_steps: int = 5000,
) -> tuple:
    """Build synthetic time series by filtering and repeating qualifying voyages.

    Selects voyage segments with energy <= max_voyage_energy, then
    concatenates them (with port stays between) to create a synthetic
    route profile representing short-haul or coastal operations.

    Args:
        p_battery_kw: Original power demand time series.
        speed: Original speed time series.
        segments: Voyage segments from identify_voyage_segments().
        segment_energies: Energy per segment (kWh).
        max_voyage_energy: Maximum voyage energy to include.
        min_total_steps: Minimum total timesteps in output (repeat if needed).

    Returns:
        (new_p_battery, new_speed, n_qualifying_voyages, period_days)
    """
    # Find qualifying voyages and their surrounding port stays
    qualifying_idx = np.where(segment_energies <= max_voyage_energy)[0]

    if len(qualifying_idx) == 0:
        raise ValueError(f"No voyages found with energy <= {max_voyage_energy} kWh")

    # Build blocks: port_stay + voyage for each qualifying segment
    blocks_p = []
    blocks_speed = []
    port_duration_steps = 12  # 1 hour default port stay (12 × 5min)

    for idx in qualifying_idx:
        seg = segments[idx]
        voyage_start = seg["start"]
        voyage_end = seg["end"] + 1

        # Always synthesize a clean port stay (speed=0, power=0)
        # to ensure proper charging between voyages
        blocks_p.append(np.full(port_duration_steps, 0.0))
        blocks_speed.append(np.full(port_duration_steps, 0.0))

        # Voyage data
        blocks_p.append(p_battery_kw[voyage_start:voyage_end])
        blocks_speed.append(speed[voyage_start:voyage_end])

    # Concatenate
    new_p = np.concatenate(blocks_p)
    new_speed = np.concatenate(blocks_speed)

    # Add trailing port stay
    new_p = np.concatenate([new_p, np.zeros(port_duration_steps)])
    new_speed = np.concatenate([new_speed, np.zeros(port_duration_steps)])

    # Repeat if too short
    if len(new_p) < min_total_steps:
        repeats = int(np.ceil(min_total_steps / len(new_p)))
        new_p = np.tile(new_p, repeats)[:min_total_steps]
        new_speed = np.tile(new_speed, repeats)[:min_total_steps]

    # Calculate period in days
    period_days = len(new_p) * (5 / 60) / 24

    return new_p, new_speed, len(qualifying_idx), period_days


def identify_voyage_segments(speed: np.ndarray, threshold: float = 0.5) -> list:
    """Identify continuous cruising segments between port stays.

    A voyage segment is a continuous period where speed > threshold.

    Args:
        speed: Ship speed time series (knots).
        threshold: Speed threshold for cruising (knots).

    Returns:
        List of dicts with 'start', 'end' indices and 'energy_steps' count.
    """
    is_cruising = speed > threshold
    segments = []
    in_segment = False
    start = 0

    for i in range(len(is_cruising)):
        if is_cruising[i] and not in_segment:
            start = i
            in_segment = True
        elif not is_cruising[i] and in_segment:
            segments.append({"start": start, "end": i - 1, "n_steps": i - start})
            in_segment = False

    # Handle segment that runs to end
    if in_segment:
        segments.append({"start": start, "end": len(speed) - 1, "n_steps": len(speed) - start})

    return segments


def compute_voyage_energy_actual(
    p_battery_kw: np.ndarray, segments: list
) -> np.ndarray:
    """Compute actual energy demand for each voyage segment.

    Used as "perfect forecast" for weather-adaptive strategy.

    Args:
        p_battery_kw: Power demand time series (kW).
        segments: Voyage segments from identify_voyage_segments().

    Returns:
        Array of energy demand (kWh) per segment.
    """
    energies = np.zeros(len(segments))
    for i, seg in enumerate(segments):
        energies[i] = np.sum(p_battery_kw[seg["start"]:seg["end"] + 1]) * INTERVAL_HOURS
    return energies


def build_adaptive_target_map(
    speed: np.ndarray,
    p_battery_kw: np.ndarray,
    segments: list,
    segment_energies: np.ndarray,
    capacity_kwh: float,
    soc_min: float,
    soc_max: float,
    safety_margin: float = 0.15,
) -> np.ndarray:
    """Build per-timestep target SOC for weather-adaptive charging.

    For each port stay, find the next voyage segment and compute
    the target SOC needed to cover its energy demand + safety margin.

    Args:
        speed: Ship speed time series.
        p_battery_kw: Power demand time series.
        segments: Voyage segments.
        segment_energies: Energy per segment (kWh).
        capacity_kwh: Battery capacity (kWh).
        soc_min: Minimum SOC.
        soc_max: Maximum SOC.
        safety_margin: Extra margin above predicted demand.

    Returns:
        Array of target SOC for each timestep (only meaningful during charging).
    """
    n = len(speed)
    target_soc = np.full(n, soc_max)  # default to soc_max

    # Map each port-stay timestep to the next voyage segment
    seg_idx = 0
    for i in range(n):
        if speed[i] <= 0.5:  # port stay
            # Find next voyage segment
            while seg_idx < len(segments) and segments[seg_idx]["start"] <= i:
                seg_idx_temp = seg_idx + 1
                if seg_idx_temp >= len(segments):
                    break
                seg_idx = seg_idx_temp

            if seg_idx < len(segments):
                energy_needed = segment_energies[seg_idx] * (1.0 + safety_margin)
                soc_needed = soc_min + energy_needed / capacity_kwh
                target_soc[i] = np.clip(soc_needed, soc_min, soc_max)
            else:
                target_soc[i] = soc_max  # last port, charge full
        # Reset seg_idx search for next port stay
        if speed[i] > 0.5 and seg_idx > 0:
            pass  # keep current seg_idx

    return target_soc


def simulate_soc(
    p_battery_kw: np.ndarray,
    speed: np.ndarray,
    chemistry: str = "NMC",
    initial_soc: float = 0.90,
    strategy: str = "fixed_full",
    capacity_kwh: float = None,
    adaptive_target_soc: np.ndarray = None,
) -> dict:
    """Simulate battery SOC over time with configurable charging strategy.

    Args:
        p_battery_kw: Power demand time series (kW).
        speed: Ship speed time series (knots).
        chemistry: Battery chemistry ("NMC" or "LFP").
        initial_soc: Starting SOC (0-1).
        strategy: "fixed_full", "conservative", or "weather_adaptive".
        capacity_kwh: Override battery capacity (kWh). None uses default.
        adaptive_target_soc: Per-timestep target SOC for adaptive strategy.

    Returns:
        dict with soc, cycles, charge/discharge info, and SOC zone statistics.
    """
    spec = BATTERY_SPECS[chemistry]
    capacity = capacity_kwh if capacity_kwh is not None else spec["capacity_kWh"]
    soc_min = spec["soc_min"]
    soc_max = spec["soc_max"]
    # Scale charge rate proportionally to capacity (0.4C rate)
    # Default 5000 kWh → 2000 kW = 0.4C. Maintain same C-rate for larger batteries.
    base_capacity = spec["capacity_kWh"]
    charge_rate = spec["charge_rate_kW"] * (capacity / base_capacity)
    eta_discharge = spec["discharge_efficiency"]
    eta_charge = spec["charge_efficiency"]

    # Determine charge target based on strategy
    if strategy == "conservative":
        charge_target = min(CHARGING_STRATEGIES["conservative"]["target_soc"], soc_max)
    elif strategy == "fixed_full":
        charge_target = soc_max
    # weather_adaptive uses adaptive_target_soc per timestep

    n = len(p_battery_kw)
    soc = np.zeros(n)
    soc[0] = initial_soc

    is_charging = np.zeros(n, dtype=bool)
    is_discharging = np.zeros(n, dtype=bool)

    # Track individual discharge events for DoD calculation
    depth_of_discharge_accum = 0.0
    discharge_events = []  # list of (soc_start, soc_end) per voyage
    current_discharge_start = initial_soc

    for i in range(1, n):
        if speed[i] > 0.5:
            # Cruising → discharge
            energy_demand = p_battery_kw[i] * INTERVAL_HOURS / eta_discharge
            delta_soc = energy_demand / capacity
            new_soc = soc[i - 1] - delta_soc

            soc[i] = max(new_soc, soc_min)
            is_discharging[i] = True

            if soc[i] < soc[i - 1]:
                depth_of_discharge_accum += soc[i - 1] - soc[i]

        else:
            # Stationary → charge from shore power
            # Determine target SOC for this timestep
            if strategy == "weather_adaptive" and adaptive_target_soc is not None:
                step_target = adaptive_target_soc[i]
            elif strategy == "conservative":
                step_target = charge_target
            else:  # fixed_full
                step_target = soc_max

            # Only charge if below target
            if soc[i - 1] < step_target:
                energy_charge = charge_rate * INTERVAL_HOURS * eta_charge
                delta_soc = energy_charge / capacity
                new_soc = soc[i - 1] + delta_soc
                soc[i] = min(new_soc, step_target)
            else:
                soc[i] = soc[i - 1]

            is_charging[i] = True

    # Equivalent full cycles
    usable_range = soc_max - soc_min
    equivalent_cycles = depth_of_discharge_accum / usable_range

    # Compute per-cycle DoD statistics
    # Identify individual discharge cycles (port→cruise→port)
    dod_list = _compute_cycle_dods(soc, is_discharging)

    return {
        "soc": soc,
        "is_charging": is_charging,
        "is_discharging": is_discharging,
        "equivalent_cycles": equivalent_cycles,
        "depth_of_discharge_total": depth_of_discharge_accum,
        "chemistry": chemistry,
        "strategy": strategy,
        "capacity_kwh": capacity,
        "cycle_dods": dod_list,
    }


def _compute_cycle_dods(soc: np.ndarray, is_discharging: np.ndarray) -> list:
    """Extract individual cycle depths of discharge.

    A "cycle" starts when discharging begins and ends when charging begins.

    Returns:
        List of DoD values for each discharge event.
    """
    dods = []
    in_discharge = False
    cycle_max_soc = 0.0
    cycle_min_soc = 1.0

    for i in range(len(soc)):
        if is_discharging[i]:
            if not in_discharge:
                # Start of new discharge event
                cycle_max_soc = soc[i - 1] if i > 0 else soc[i]
                cycle_min_soc = soc[i]
                in_discharge = True
            else:
                cycle_min_soc = min(cycle_min_soc, soc[i])
        else:
            if in_discharge:
                # End of discharge event
                dod = cycle_max_soc - cycle_min_soc
                if dod > 0.01:  # filter noise
                    dods.append(dod)
                in_discharge = False

    # Handle trailing discharge
    if in_discharge:
        dod = cycle_max_soc - cycle_min_soc
        if dod > 0.01:
            dods.append(dod)

    return dods


def compute_soc_statistics(soc_result: dict) -> dict:
    """Compute summary statistics from SOC simulation.

    Includes SOC zone statistics for degradation analysis.
    """
    soc = soc_result["soc"]
    dods = soc_result.get("cycle_dods", [])

    return {
        "chemistry": soc_result["chemistry"],
        "strategy": soc_result.get("strategy", "fixed_full"),
        "capacity_kwh": soc_result.get("capacity_kwh", 5000),
        "mean_soc": soc.mean(),
        "min_soc": soc.min(),
        "max_soc": soc.max(),
        "std_soc": soc.std(),
        "equivalent_cycles": soc_result["equivalent_cycles"],
        "charge_time_pct": soc_result["is_charging"].mean() * 100,
        "discharge_time_pct": soc_result["is_discharging"].mean() * 100,
        # SOC zone statistics
        "time_at_high_soc_pct": (soc > 0.80).mean() * 100,
        "time_at_low_soc_pct": (soc < 0.20).mean() * 100,
        "time_in_sweet_spot_pct": ((soc >= 0.20) & (soc <= 0.80)).mean() * 100,
        "soc_floor_hits": int((soc <= soc_result.get("soc_min_used", 0.15) + 0.01).sum()),
        # DoD statistics
        "mean_dod": np.mean(dods) if dods else 0.0,
        "max_dod": np.max(dods) if dods else 0.0,
        "n_cycles_counted": len(dods),
    }
