"""Tests for data preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    WEATHER_FEATURES,
    SPEED_FEATURE,
    TARGET,
    derive_operating_state,
    select_features,
    time_based_split,
    create_scaled_datasets,
    create_sequence_dataset,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking Poseidon structure."""
    n = 200
    rng = np.random.RandomState(42)
    data = {
        "Ship_SpeedOverGround": rng.uniform(0, 12, n),
        "P_battery_kW": rng.uniform(3000, 50000, n),
    }
    # Add all 26 weather features
    for feat in WEATHER_FEATURES:
        data[feat] = rng.uniform(0, 100, n)
    # Add some NaNs
    data["Ship_SpeedOverGround"][5] = np.nan
    data["Weather_WindSpeed10M"][10] = np.nan
    return pd.DataFrame(data)


class TestOperatingState:
    def test_stationary(self):
        speeds = pd.Series([0.0, 0.1, 0.4])
        result = derive_operating_state(speeds)
        assert (result == 0).all()

    def test_maneuvering(self):
        speeds = pd.Series([0.5, 1.0, 1.9])
        result = derive_operating_state(speeds)
        assert (result == 1).all()

    def test_cruising(self):
        speeds = pd.Series([2.0, 5.0, 12.0])
        result = derive_operating_state(speeds)
        assert (result == 2).all()

    def test_nan_handled(self):
        speeds = pd.Series([np.nan, 5.0])
        result = derive_operating_state(speeds)
        assert result.isna().sum() <= 1  # NaN propagated or filled


class TestSelectFeatures:
    def test_output_columns(self, sample_df):
        result = select_features(sample_df)
        assert TARGET in result.columns
        assert SPEED_FEATURE in result.columns
        assert "operating_state" in result.columns
        assert len(result.columns) == len(WEATHER_FEATURES) + 3  # weather + speed + state + target

    def test_no_nans_after_processing(self, sample_df):
        result = select_features(sample_df)
        assert result.isna().sum().sum() == 0


class TestTimeBasedSplit:
    def test_split_ratios(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert abs(len(train) / len(df) - 0.7) < 0.02
        assert abs(len(val) / len(df) - 0.15) < 0.02

    def test_no_overlap(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        # After reset_index, verify lengths partition correctly
        assert len(train) + len(val) + len(test) == len(df)
        # Verify chronological order by checking original values
        assert train.iloc[-1][TARGET] == df.iloc[len(train) - 1][TARGET]
        assert val.iloc[0][TARGET] == df.iloc[len(train)][TARGET]


class TestScaledDatasets:
    def test_output_shapes(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        datasets, scaler_X, scaler_y = create_scaled_datasets(train, val, test)
        X_train, y_train = datasets["train"]
        X_val, y_val = datasets["val"]
        X_test, y_test = datasets["test"]
        n_features = len(WEATHER_FEATURES) + 2  # weather + speed + state
        assert X_train.shape[1] == n_features
        assert len(y_train.shape) == 1

    def test_train_mean_near_zero(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        datasets, _, _ = create_scaled_datasets(train, val, test)
        X_train = datasets["train"][0]
        assert abs(X_train.mean()) < 0.5  # roughly centered


class TestSequenceDataset:
    def test_sequence_shape(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        datasets, _, _ = create_scaled_datasets(train, val, test)
        X_train, y_train = datasets["train"]
        seq_X, seq_y = create_sequence_dataset(X_train, y_train, seq_len=24)
        assert seq_X.shape[0] == len(X_train) - 24
        assert seq_X.shape[1] == 24
        assert seq_X.shape[2] == X_train.shape[1]
        assert seq_y.shape[0] == seq_X.shape[0]

    def test_sequence_values(self, sample_df):
        df = select_features(sample_df)
        train, val, test = time_based_split(df)
        datasets, _, _ = create_scaled_datasets(train, val, test)
        X_train, y_train = datasets["train"]
        seq_X, seq_y = create_sequence_dataset(X_train, y_train, seq_len=4)
        # First sequence should match first 4 rows
        np.testing.assert_array_almost_equal(seq_X[0], X_train[:4])
        # Target should be the value after the sequence (float32 tolerance)
        np.testing.assert_almost_equal(seq_y[0], y_train[4], decimal=5)
