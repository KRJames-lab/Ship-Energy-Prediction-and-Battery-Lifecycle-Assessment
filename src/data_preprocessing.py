"""Data preprocessing for Phase 2: feature selection, splitting, scaling, sequencing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WEATHER_FEATURES = [
    "Weather_DiffuseRadiation",
    "Weather_DirectNormalIrradiance",
    "Weather_DirectRadiation",
    "Weather_OceanCurrentDirection",
    "Weather_OceanCurrentVelocity",
    "Weather_Precipitation",
    "Weather_RelativeHumidity2M",
    "Weather_ShortwaveRadiation",
    "Weather_SunshineDuration",
    "Weather_SurfacePressure",
    "Weather_SwellWaveDirection",
    "Weather_SwellWaveHeight",
    "Weather_SwellWavePeakPeriod",
    "Weather_SwellWavePeriod",
    "Weather_Temperature2M",
    "Weather_WaveDirection",
    "Weather_WaveHeight",
    "Weather_WavePeriod",
    "Weather_WeatherCode",
    "Weather_WindDirection10M",
    "Weather_WindGusts10M",
    "Weather_WindSpeed10M",
    "Weather_WindWaveDirection",
    "Weather_WindWaveHeight",
    "Weather_WindWavePeakPeriod",
    "Weather_WindWavePeriod",
]

SPEED_FEATURE = "Ship_SpeedOverGround"
TARGET = "P_battery_kW"

INPUT_FEATURES = WEATHER_FEATURES + [SPEED_FEATURE, "operating_state"]


def derive_operating_state(speed: pd.Series) -> pd.Series:
    """Derive operating state from speed: 0=stationary, 1=maneuvering, 2=cruising."""
    state = pd.Series(np.nan, index=speed.index)
    state[speed < 0.5] = 0
    state[(speed >= 0.5) & (speed < 2.0)] = 1
    state[speed >= 2.0] = 2
    return state


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and preprocess features, interpolate missing values."""
    result = df[WEATHER_FEATURES + [SPEED_FEATURE, TARGET]].copy()

    # Interpolate then fill edges
    result = result.interpolate(method="linear").ffill().bfill()

    result["operating_state"] = derive_operating_state(result[SPEED_FEATURE])
    result["operating_state"] = result["operating_state"].ffill().bfill().fillna(0)

    return result


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple:
    """Split DataFrame chronologically into train/val/test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    return train, val, test


def create_scaled_datasets(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple:
    """Scale features using StandardScaler fitted on train set only.

    Returns:
        (datasets_dict, scaler_X, scaler_y) where datasets_dict maps
        split name to (X_array, y_array) tuples.
    """
    feature_cols = INPUT_FEATURES

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train[feature_cols].values)
    y_train = scaler_y.fit_transform(train[[TARGET]].values).ravel()

    X_val = scaler_X.transform(val[feature_cols].values)
    y_val = scaler_y.transform(val[[TARGET]].values).ravel()

    X_test = scaler_X.transform(test[feature_cols].values)
    y_test = scaler_y.transform(test[[TARGET]].values).ravel()

    datasets = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    return datasets, scaler_X, scaler_y


def create_sequence_dataset(
    X: np.ndarray, y: np.ndarray, seq_len: int = 24
) -> tuple:
    """Create sliding window sequences for LSTM/Transformer.

    Args:
        X: Feature array (n_samples, n_features).
        y: Target array (n_samples,).
        seq_len: Number of past time steps per sequence.

    Returns:
        (X_seq, y_seq) where X_seq is (n_sequences, seq_len, n_features)
        and y_seq is (n_sequences,) predicting the next step.
    """
    n = len(X) - seq_len
    n_features = X.shape[1]

    X_seq = np.zeros((n, seq_len, n_features), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)

    for i in range(n):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len]

    return X_seq, y_seq
