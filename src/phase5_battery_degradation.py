"""Phase 5: Arrhenius Battery Degradation with SOC-Dependent Stress.

Models battery State of Health (SOH) degradation over 10 years:
- Calendar aging: time + temperature + SOC-dependent (Arrhenius)
- Cycle aging: charge/discharge cycle + DoD-dependent
- Compares NMC vs LFP, cold (3°C) vs tropical (25°C)

SOC-Dependent Degradation (literature-based):
- High SOC (>80%): accelerated calendar aging due to cathode oxidation
- Low SOC (<20%): accelerated cycle aging due to anode stress
- Deep DoD: cycle aging scales as DoD^beta (beta ≈ 1.5-2.0)
"""
import numpy as np

# Arrhenius parameters (empirical, from literature)
R_GAS = 8.314  # J/(mol·K)

DEGRADATION_PARAMS = {
    "NMC": {
        "Ea_cal": 35_000,       # Activation energy, calendar (J/mol)
        "k_cal": 4.3e4,         # Calibrated: ~10% calendar loss at 25°C/10y
        "k_cyc": 0.0008,        # Cycle aging: ~83% SOH at 25°C/10y with marine cycling
        "cyc_exponent": 0.5,
        # SOC-dependent parameters
        "alpha_soc": 0.8,       # Calendar aging SOC stress factor
        "beta_dod": 1.6,        # DoD exponent for cycle aging
    },
    "LFP": {
        "Ea_cal": 40_000,       # Higher Ea → slower calendar aging
        "k_cal": 1.6e5,         # Calibrated: ~5% calendar loss at 25°C/10y
        "k_cyc": 0.0003,        # Much slower cycle aging
        "cyc_exponent": 0.5,
        # SOC-dependent parameters
        "alpha_soc": 0.5,       # LFP less sensitive to SOC stress
        "beta_dod": 1.4,        # LFP more tolerant of deep DoD
    },
}


def soc_stress_factor(mean_soc: float, alpha: float) -> float:
    """Compute SOC-dependent calendar aging stress factor.

    Higher SOC accelerates calendar aging due to cathode oxidation
    and electrolyte decomposition.

    k_soc = exp(alpha * (mean_soc - 0.5))

    At SOC=0.5: k_soc = 1.0 (neutral)
    At SOC=0.95: k_soc ≈ 1.43 (NMC, alpha=0.8) → 43% faster calendar aging
    At SOC=0.70: k_soc ≈ 1.17 (NMC) → 17% faster
    At SOC=0.50: k_soc = 1.0 (optimal)

    Args:
        mean_soc: Average SOC during operation (0-1).
        alpha: SOC sensitivity coefficient.

    Returns:
        Stress multiplier (>1 accelerates, <1 decelerates).
    """
    return np.exp(alpha * (mean_soc - 0.5))


def dod_stress_factor(mean_dod: float, beta: float) -> float:
    """Compute DoD-dependent cycle aging stress factor.

    Deeper discharge cycles cause more damage per cycle.

    k_dod = mean_dod^beta / reference_dod^beta

    Reference DoD = 0.8 (standard full cycle test condition).
    Shallow cycles (DoD=0.2) → k_dod ≈ 0.11 (NMC) → 89% less damage per cycle
    Full cycles (DoD=0.8) → k_dod = 1.0 (reference)
    Deep cycles (DoD=0.9) → k_dod ≈ 1.20 → 20% more damage

    Args:
        mean_dod: Average depth of discharge per cycle (0-1).
        beta: DoD sensitivity exponent.

    Returns:
        Stress multiplier relative to reference DoD.
    """
    reference_dod = 0.8
    if mean_dod < 0.01:
        return 0.0
    return (mean_dod / reference_dod) ** beta


def calendar_aging(
    t_years: np.ndarray, temp_celsius: float, chemistry: str,
    mean_soc: float = None,
) -> np.ndarray:
    """Compute calendar aging SOH loss with optional SOC stress.

    Uses Arrhenius equation: loss = k * exp(-Ea/(R*T)) * sqrt(t) * k_soc

    Args:
        t_years: Time array in years.
        temp_celsius: Ambient temperature (°C).
        chemistry: "NMC" or "LFP".
        mean_soc: Average SOC (enables SOC-dependent aging if provided).

    Returns:
        SOH loss fraction (0 to 1) at each time point.
    """
    params = DEGRADATION_PARAMS[chemistry]
    T_kelvin = temp_celsius + 273.15

    rate = params["k_cal"] * np.exp(-params["Ea_cal"] / (R_GAS * T_kelvin))

    # Apply SOC stress factor
    if mean_soc is not None:
        k_soc = soc_stress_factor(mean_soc, params["alpha_soc"])
        rate *= k_soc

    loss = rate * np.sqrt(t_years + 1e-10)  # avoid sqrt(0)

    return np.minimum(loss, 1.0)


def cycle_aging(
    n_cycles: np.ndarray, chemistry: str,
    mean_dod: float = None,
) -> np.ndarray:
    """Compute cycle aging SOH loss with optional DoD stress.

    loss = k_cyc * k_dod * N^exponent

    Args:
        n_cycles: Cumulative cycle count array.
        chemistry: "NMC" or "LFP".
        mean_dod: Average DoD per cycle (enables DoD-dependent aging).

    Returns:
        SOH loss fraction at each cycle count.
    """
    params = DEGRADATION_PARAMS[chemistry]

    k_dod = 1.0
    if mean_dod is not None:
        k_dod = dod_stress_factor(mean_dod, params["beta_dod"])

    loss = params["k_cyc"] * k_dod * (n_cycles ** params["cyc_exponent"])
    return np.minimum(loss, 1.0)


def simulate_degradation(
    cycles_per_period: float,
    period_days: int,
    temp_celsius: float,
    chemistry: str,
    years: int = 10,
    mean_soc: float = None,
    mean_dod: float = None,
) -> dict:
    """Simulate 10-year battery degradation with SOC/DoD stress factors.

    Extrapolates short-term cycling data to long-term SOH curve
    by repeating the measured cycle pattern.

    Args:
        cycles_per_period: Equivalent full cycles in the data period.
        period_days: Duration of the data period (days).
        temp_celsius: Average ambient temperature (°C).
        chemistry: "NMC" or "LFP".
        years: Simulation horizon.
        mean_soc: Average SOC from simulation (for SOC stress).
        mean_dod: Average DoD per cycle (for DoD stress).

    Returns:
        dict with time_years, soh, soh_calendar, soh_cycle arrays.
    """
    # Monthly time steps
    n_months = years * 12
    t_years = np.linspace(0, years, n_months + 1)

    # Extrapolate cycles: cycles/day * 365 * years
    cycles_per_day = cycles_per_period / period_days
    cumulative_cycles = cycles_per_day * t_years * 365

    # Calendar and cycle aging (with stress factors)
    loss_cal = calendar_aging(t_years, temp_celsius, chemistry, mean_soc=mean_soc)
    loss_cyc = cycle_aging(cumulative_cycles, chemistry, mean_dod=mean_dod)

    # Total SOH = 1 - (calendar_loss + cycle_loss), clamped
    soh = np.maximum(1.0 - loss_cal - loss_cyc, 0.0)

    # Find year when SOH reaches 80% (end of life)
    eol_mask = soh <= 0.80
    eol_year = t_years[eol_mask][0] if eol_mask.any() else None

    # SOC/DoD stress info
    params = DEGRADATION_PARAMS[chemistry]
    k_soc_val = soc_stress_factor(mean_soc, params["alpha_soc"]) if mean_soc is not None else 1.0
    k_dod_val = dod_stress_factor(mean_dod, params["beta_dod"]) if mean_dod is not None else 1.0

    return {
        "chemistry": chemistry,
        "temp_celsius": temp_celsius,
        "time_years": t_years,
        "soh": soh,
        "soh_calendar_loss": loss_cal,
        "soh_cycle_loss": loss_cyc,
        "cumulative_cycles": cumulative_cycles,
        "eol_year_80pct": eol_year,
        "mean_soc": mean_soc,
        "mean_dod": mean_dod,
        "k_soc_stress": k_soc_val,
        "k_dod_stress": k_dod_val,
    }


def simulate_degradation_stepwise(
    soc_timeseries: np.ndarray,
    cycle_dods: list,
    period_days: float,
    temp_celsius: float,
    chemistry: str,
    years: int = 10,
    dt_hours: float = 5 / 60,
) -> dict:
    """Simulate 10-year degradation using step-wise SOC-dependent aging.

    Instead of using mean SOC as a constant, this computes calendar aging
    at every 5-minute step based on the actual SOC at that moment.
    The 3-month SOC pattern is tiled to cover the full simulation horizon.

    Calendar aging: at each step, d_loss = base_rate × k_soc(SOC[t]) × d(√t)
    Cycle aging: per-cycle DoD damage, accumulated with √N scaling.

    Args:
        soc_timeseries: SOC array from Phase 4 (5-min intervals).
        cycle_dods: List of DoD values per discharge cycle from Phase 4.
        period_days: Duration of the SOC data (days).
        temp_celsius: Ambient temperature (°C).
        chemistry: "NMC" or "LFP".
        years: Simulation horizon (default 10).
        dt_hours: Time step in hours (default 5/60).

    Returns:
        dict with time_years, soh, soh_calendar_loss, soh_cycle_loss arrays
        sampled at monthly intervals.
    """
    params = DEGRADATION_PARAMS[chemistry]
    T_kelvin = temp_celsius + 273.15
    base_rate = params["k_cal"] * np.exp(-params["Ea_cal"] / (R_GAS * T_kelvin))
    dt_years = dt_hours / (365 * 24)

    # Tile SOC pattern to cover 10 years
    steps_per_period = len(soc_timeseries)
    total_steps = int(years * 365 * 24 / (dt_hours))
    n_repeats = int(np.ceil(total_steps / steps_per_period))
    soc_full = np.tile(soc_timeseries, n_repeats)[:total_steps]

    # --- Calendar aging: step-wise with actual SOC ---
    # d(√t) = √(t+dt) - √(t) at each step
    t_array = np.arange(total_steps, dtype=np.float64) * dt_years
    sqrt_t_next = np.sqrt(t_array + dt_years)
    sqrt_t_curr = np.sqrt(t_array + 1e-12)
    sqrt_increments = sqrt_t_next - sqrt_t_curr

    # SOC stress factor at each step: k_soc = exp(alpha * (SOC - 0.5))
    k_soc_array = np.exp(params["alpha_soc"] * (soc_full - 0.5))

    # Cumulative calendar loss
    cal_increments = base_rate * k_soc_array * sqrt_increments
    cal_loss_full = np.cumsum(cal_increments)

    # --- Cycle aging: per-cycle with actual DoDs ---
    if len(cycle_dods) > 0:
        # Tile cycle DoDs for 10 years
        all_dods = np.tile(cycle_dods, n_repeats)
        total_cycles = int(len(cycle_dods) * (years * 365 / period_days))
        all_dods = all_dods[:total_cycles]

        # Per-cycle damage with √N scaling: d_loss = k_cyc × k_dod_i × (√i - √(i-1))
        N = np.arange(1, total_cycles + 1, dtype=np.float64)
        sqrt_N_inc = np.sqrt(N) - np.sqrt(N - 1)
        k_dod_per_cycle = (np.array(all_dods) / 0.8) ** params["beta_dod"]
        cyc_damage_cumulative = np.cumsum(
            params["k_cyc"] * k_dod_per_cycle * sqrt_N_inc
        )
    else:
        total_cycles = 0
        cyc_damage_cumulative = np.array([0.0])

    # --- Sample at monthly intervals ---
    n_months = years * 12
    t_years_monthly = np.array([m / 12.0 for m in range(n_months + 1)])

    # Calendar: sample from the step-wise array
    steps_per_month = int(30 * 24 / dt_hours)
    cal_at_months = np.zeros(n_months + 1)
    for m in range(1, n_months + 1):
        idx = min(m * steps_per_month - 1, total_steps - 1)
        cal_at_months[m] = cal_loss_full[idx]

    # Cycle: map months to cumulative cycle count
    cycles_per_day = len(cycle_dods) / period_days if period_days > 0 else 0
    cyc_at_months = np.zeros(n_months + 1)
    for m in range(1, n_months + 1):
        cum_cyc = int(m * 30 * cycles_per_day)
        if cum_cyc > 0 and cum_cyc <= len(cyc_damage_cumulative):
            cyc_at_months[m] = cyc_damage_cumulative[cum_cyc - 1]
        elif cum_cyc > len(cyc_damage_cumulative) and len(cyc_damage_cumulative) > 0:
            cyc_at_months[m] = cyc_damage_cumulative[-1]

    # Total SOH
    soh = np.maximum(1.0 - cal_at_months - cyc_at_months, 0.0)

    # EOL
    eol_mask = soh <= 0.80
    eol_year = t_years_monthly[eol_mask][0] if eol_mask.any() else None

    # Summary stats
    mean_k_soc = float(k_soc_array.mean())
    mean_k_dod = float(np.mean(
        (np.array(cycle_dods) / 0.8) ** params["beta_dod"]
    )) if len(cycle_dods) > 0 else 0.0

    return {
        "chemistry": chemistry,
        "temp_celsius": temp_celsius,
        "time_years": t_years_monthly,
        "soh": soh,
        "soh_calendar_loss": cal_at_months,
        "soh_cycle_loss": cyc_at_months,
        "eol_year_80pct": eol_year,
        "mean_soc": float(soc_timeseries.mean()),
        "mean_dod": float(np.mean(cycle_dods)) if len(cycle_dods) > 0 else 0.0,
        "k_soc_stress": mean_k_soc,
        "k_dod_stress": mean_k_dod,
    }
