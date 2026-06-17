"""Phase 1: Diesel-to-Electric Energy Conversion.

Converts measured diesel fuel consumption to equivalent battery electric power (kW)
for both Triton (HVO 30) and Poseidon (multi-fuel) vessels.

Physics:
    P_battery [kW] = (m_fuel [kg/s] * Q_LHV [kJ/kg] * eta_diesel) / eta_electric
"""
import numpy as np
import pandas as pd

# --- Fuel Parameters ---
FUEL_PARAMS = {
    "HVO 30": {"Q_LHV": 43_100, "eta_diesel": 0.42},   # kJ/kg, HVO 30% bio-blend
    "DM":     {"Q_LHV": 45_500, "eta_diesel": 0.43},    # Diesel Marine (ISO 8217)
    "RM 380": {"Q_LHV": 44_100, "eta_diesel": 0.40},    # Residual Marine 380
    "RM 180": {"Q_LHV": 44_100, "eta_diesel": 0.40},    # Residual Marine 180
}

ETA_ELECTRIC = 0.91  # Motor + inverter + battery combined efficiency

# 5-minute sampling interval in hours
INTERVAL_HOURS = 5 / 60


def compute_battery_power_single_fuel(m_fuel: float, fuel_type: str) -> float:
    """Compute battery-equivalent power for a single fuel type.

    Args:
        m_fuel: Fuel mass flow rate (kg/s).
        fuel_type: Fuel type key (e.g., "HVO 30").

    Returns:
        Battery power in kW.
    """
    params = FUEL_PARAMS[fuel_type]
    return (m_fuel * params["Q_LHV"] * params["eta_diesel"]) / ETA_ELECTRIC


def compute_battery_power_multi_fuel(
    fuel_rates: list, fuel_types: list
) -> float:
    """Compute battery-equivalent power from multiple engines with different fuels.

    Args:
        fuel_rates: List of fuel mass flow rates (kg/s) per engine.
        fuel_types: List of fuel type keys per engine.

    Returns:
        Total battery power in kW.
    """
    if len(fuel_rates) != len(fuel_types):
        raise ValueError(
            f"Length mismatch: fuel_rates={len(fuel_rates)}, fuel_types={len(fuel_types)}"
        )
    total_diesel_kw = sum(
        rate * FUEL_PARAMS[ftype]["Q_LHV"] * FUEL_PARAMS[ftype]["eta_diesel"]
        for rate, ftype in zip(fuel_rates, fuel_types)
    )
    return total_diesel_kw / ETA_ELECTRIC


def convert_triton(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Triton diesel fuel consumption to battery-equivalent power.

    Args:
        df: Triton DataFrame with Consumer_Total_MomentaryFuel column.

    Returns:
        DataFrame with added P_battery_kW, P_diesel_kW, energy_kWh columns.
    """
    result = df.copy()

    m_fuel = result["Consumer_Total_MomentaryFuel"].interpolate(method="linear")
    m_fuel = m_fuel.fillna(0.0)  # fill edge NaNs

    params = FUEL_PARAMS["HVO 30"]
    result["P_diesel_kW"] = m_fuel * params["Q_LHV"] * params["eta_diesel"]
    result["P_battery_kW"] = result["P_diesel_kW"] / ETA_ELECTRIC
    result["energy_kWh"] = result["P_battery_kW"] * INTERVAL_HOURS

    return result


# Poseidon engine definitions: (fuel_col, fuel_type_col) pairs
_POSEIDON_ENGINES = [
    ("Consumer_Boiler1_MomentaryFuel", "Consumer_Boiler1_FuelType"),
    ("Consumer_Boiler2_MomentaryFuel", "Consumer_Boiler2_FuelType"),
    ("Consumer_GeneratorEngine1_MomentaryFuel", "Consumer_GeneratorEngine1_FuelType"),
    ("Consumer_GeneratorEngine2_MomentaryFuel", "Consumer_GeneratorEngine2_FuelType"),
    ("Consumer_GeneratorEngine3_MomentaryFuel", "Consumer_GeneratorEngine3_FuelType"),
    ("Consumer_GeneratorEngine4_MomentaryFuel", "Consumer_GeneratorEngine4_FuelType"),
    ("Consumer_GeneratorEngine5_MomentaryFuel", "Consumer_GeneratorEngine5_FuelType"),
]


def convert_poseidon(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Poseidon multi-fuel diesel consumption to battery-equivalent power.

    Calculates per-engine diesel power using each engine's fuel type,
    then sums and divides by electric efficiency.

    Args:
        df: Poseidon DataFrame with per-engine fuel columns.

    Returns:
        DataFrame with added P_battery_kW, P_diesel_kW, energy_kWh columns.
    """
    result = df.copy()

    total_diesel_kw = np.zeros(len(result))

    for fuel_col, ftype_col in _POSEIDON_ENGINES:
        m_fuel = result[fuel_col].interpolate(method="linear").fillna(0.0)

        for ftype, params in FUEL_PARAMS.items():
            mask = result[ftype_col] == ftype
            total_diesel_kw[mask] += (
                m_fuel[mask].values * params["Q_LHV"] * params["eta_diesel"]
            )

    result["P_diesel_kW"] = total_diesel_kw
    result["P_battery_kW"] = total_diesel_kw / ETA_ELECTRIC
    result["energy_kWh"] = result["P_battery_kW"] * INTERVAL_HOURS

    return result
