"""Phase 1: Diesel-to-Electric Energy Conversion Tests."""
import numpy as np
import pandas as pd
import pytest

from src.phase1_energy_conversion import (
    FUEL_PARAMS,
    ETA_ELECTRIC,
    compute_battery_power_single_fuel,
    compute_battery_power_multi_fuel,
    convert_triton,
    convert_poseidon,
)


class TestFuelParameters:
    """Validate fuel parameter definitions."""

    def test_all_fuel_types_defined(self):
        assert "HVO 30" in FUEL_PARAMS
        assert "DM" in FUEL_PARAMS
        assert "RM 380" in FUEL_PARAMS
        assert "RM 180" in FUEL_PARAMS

    def test_q_lhv_ranges(self):
        for name, params in FUEL_PARAMS.items():
            assert 40_000 <= params["Q_LHV"] <= 50_000, f"{name} Q_LHV out of range"

    def test_eta_diesel_ranges(self):
        for name, params in FUEL_PARAMS.items():
            assert 0.30 <= params["eta_diesel"] <= 0.50, f"{name} eta_diesel out of range"

    def test_eta_electric_range(self):
        assert 0.85 <= ETA_ELECTRIC <= 0.95


class TestSingleFuelConversion:
    """Test single-fuel conversion (Triton-like)."""

    def test_zero_fuel_gives_zero_power(self):
        result = compute_battery_power_single_fuel(0.0, "HVO 30")
        assert result == 0.0

    def test_positive_fuel_gives_positive_power(self):
        result = compute_battery_power_single_fuel(0.176, "HVO 30")
        assert result > 0.0

    def test_known_value_triton(self):
        # m_fuel=0.176 kg/s, Q_LHV=43100 kJ/kg, eta_d=0.42, eta_e=0.91
        # P = (0.176 * 43100 * 0.42) / 0.91 = 3501.7 kW (approx)
        expected = (0.176 * 43100 * 0.42) / 0.91
        result = compute_battery_power_single_fuel(0.176, "HVO 30")
        assert abs(result - expected) < 0.1

    def test_linearity(self):
        p1 = compute_battery_power_single_fuel(0.1, "HVO 30")
        p2 = compute_battery_power_single_fuel(0.2, "HVO 30")
        assert abs(p2 / p1 - 2.0) < 1e-6


class TestMultiFuelConversion:
    """Test multi-fuel conversion (Poseidon-like)."""

    def test_single_engine(self):
        fuel_rates = [0.1]
        fuel_types = ["DM"]
        result = compute_battery_power_multi_fuel(fuel_rates, fuel_types)
        expected = (0.1 * 45500 * 0.43) / 0.91
        assert abs(result - expected) < 0.1

    def test_mixed_fuels(self):
        fuel_rates = [0.1, 0.2]
        fuel_types = ["DM", "RM 380"]
        result = compute_battery_power_multi_fuel(fuel_rates, fuel_types)
        expected = (0.1 * 45500 * 0.43 + 0.2 * 44100 * 0.40) / 0.91
        assert abs(result - expected) < 0.1

    def test_all_zero_fuel(self):
        result = compute_battery_power_multi_fuel([0.0, 0.0], ["DM", "RM 380"])
        assert result == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_battery_power_multi_fuel([0.1], ["DM", "RM 380"])


class TestTritonConversion:
    """Integration test for full Triton dataset conversion."""

    @pytest.fixture
    def triton_df(self):
        return pd.DataFrame({
            "Consumer_Total_MomentaryFuel": [0.0, 0.1, 0.2, np.nan, 0.15],
            "Consumer_Boiler_FuelType": ["HVO 30"] * 5,
            "Ship_SpeedOverGround": [0.0, 5.0, 8.0, 6.0, 7.0],
        })

    def test_output_columns_exist(self, triton_df):
        result = convert_triton(triton_df)
        assert "P_battery_kW" in result.columns
        assert "P_diesel_kW" in result.columns
        assert "energy_kWh" in result.columns

    def test_no_negative_power(self, triton_df):
        result = convert_triton(triton_df)
        assert (result["P_battery_kW"] >= 0).all()

    def test_nan_interpolated(self, triton_df):
        result = convert_triton(triton_df)
        assert result["P_battery_kW"].isna().sum() == 0

    def test_zero_fuel_zero_power(self, triton_df):
        result = convert_triton(triton_df)
        assert result.loc[0, "P_battery_kW"] == 0.0

    def test_energy_kwh_calculation(self, triton_df):
        result = convert_triton(triton_df)
        # energy_kWh = P_battery_kW * (5 min / 60 min) for 5-min intervals
        interval_hours = 5 / 60
        expected_energy = result["P_battery_kW"].iloc[1] * interval_hours
        assert abs(result["energy_kWh"].iloc[1] - expected_energy) < 0.01


class TestPoseidonConversion:
    """Integration test for full Poseidon dataset conversion."""

    @pytest.fixture
    def poseidon_df(self):
        return pd.DataFrame({
            "Consumer_Boiler1_MomentaryFuel": [0.005, 0.01],
            "Consumer_Boiler1_FuelType": ["DM", "RM 380"],
            "Consumer_Boiler2_MomentaryFuel": [0.005, 0.01],
            "Consumer_Boiler2_FuelType": ["DM", "RM 380"],
            "Consumer_GeneratorEngine1_MomentaryFuel": [0.15, 0.20],
            "Consumer_GeneratorEngine1_FuelType": ["DM", "RM 380"],
            "Consumer_GeneratorEngine2_MomentaryFuel": [0.15, 0.20],
            "Consumer_GeneratorEngine2_FuelType": ["DM", "RM 380"],
            "Consumer_GeneratorEngine3_MomentaryFuel": [0.10, 0.15],
            "Consumer_GeneratorEngine3_FuelType": ["DM", "RM 380"],
            "Consumer_GeneratorEngine4_MomentaryFuel": [0.20, 0.25],
            "Consumer_GeneratorEngine4_FuelType": ["DM", "RM 380"],
            "Consumer_GeneratorEngine5_MomentaryFuel": [0.10, 0.15],
            "Consumer_GeneratorEngine5_FuelType": ["DM", "RM 380"],
            "Consumer_Total_MomentaryFuel": [0.71, 0.97],
            "Ship_SpeedOverGround": [5.0, 10.0],
        })

    def test_output_columns_exist(self, poseidon_df):
        result = convert_poseidon(poseidon_df)
        assert "P_battery_kW" in result.columns
        assert "P_diesel_kW" in result.columns
        assert "energy_kWh" in result.columns

    def test_no_negative_power(self, poseidon_df):
        result = convert_poseidon(poseidon_df)
        assert (result["P_battery_kW"] >= 0).all()

    def test_multi_fuel_different_from_single(self, poseidon_df):
        """Multi-fuel calc should differ from uniform DM assumption."""
        result = convert_poseidon(poseidon_df)
        # Uniform DM calc for comparison
        total_fuel = poseidon_df["Consumer_Total_MomentaryFuel"].iloc[1]
        uniform = (total_fuel * 45500 * 0.43) / 0.91
        # Row 1 uses RM 380 (lower efficiency) so should give less power
        assert result["P_battery_kW"].iloc[1] < uniform
