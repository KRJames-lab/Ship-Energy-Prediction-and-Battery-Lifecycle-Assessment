"""Physics-based data augmentation for small dataset enhancement.

Augmentation strategies that preserve physical relationships:
1. Speed perturbation with cubic power scaling (P ∝ V³)
2. Wind perturbation with quadratic resistance scaling (R ∝ V_wind²)
3. Wave perturbation with quadratic resistance scaling (R ∝ H_s²)
4. Gaussian noise injection on weather features
"""
import numpy as np
import pandas as pd

from src.data_preprocessing import WEATHER_FEATURES, SPEED_FEATURE, TARGET, INPUT_FEATURES
from src.physics_loss import IDX_SPEED, IDX_WIND_SPEED, IDX_WAVE_HEIGHT


def augment_speed_cubic(df: pd.DataFrame, n_augments: int = 1,
                        noise_std: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Augment by perturbing speed and adjusting power via P ∝ V³.

    For each sample, create augmented versions where:
    - Speed is scaled by (1 + noise), where noise ~ N(0, noise_std)
    - Power is adjusted by (V_new/V_old)³ ratio
    - Only applied to moving samples (speed > 0.5 knots)
    """
    rng = np.random.RandomState(seed)
    moving = df[df[SPEED_FEATURE] > 0.5].copy()

    if len(moving) == 0:
        return pd.DataFrame(columns=df.columns)

    augmented = []
    for i in range(n_augments):
        aug = moving.copy()
        speed_orig = aug[SPEED_FEATURE].values
        noise = 1.0 + rng.normal(0, noise_std, len(aug))
        noise = np.clip(noise, 0.5, 1.5)  # limit perturbation range

        speed_new = speed_orig * noise
        power_ratio = (speed_new / (speed_orig + 1e-8)) ** 3

        aug[SPEED_FEATURE] = speed_new
        aug[TARGET] = aug[TARGET].values * power_ratio
        augmented.append(aug)

    return pd.concat(augmented, ignore_index=True)


def augment_weather_noise(df: pd.DataFrame, n_augments: int = 1,
                          noise_std: float = 0.05, seed: int = 43) -> pd.DataFrame:
    """Add small Gaussian noise to weather features.

    Preserves physical plausibility by keeping noise small and
    clipping non-negative features (radiation, wave height, wind speed).
    """
    rng = np.random.RandomState(seed)
    non_negative_features = [
        "Weather_DiffuseRadiation", "Weather_DirectNormalIrradiance",
        "Weather_DirectRadiation", "Weather_ShortwaveRadiation",
        "Weather_Precipitation", "Weather_SunshineDuration",
        "Weather_SwellWaveHeight", "Weather_WaveHeight",
        "Weather_WindWaveHeight", "Weather_WindSpeed10M",
        "Weather_WindGusts10M", "Weather_OceanCurrentVelocity",
    ]

    augmented = []
    for i in range(n_augments):
        aug = df.copy()
        for feat in WEATHER_FEATURES:
            vals = aug[feat].values
            noise = rng.normal(0, noise_std * (np.abs(vals).mean() + 1e-6), len(aug))
            aug[feat] = vals + noise
            if feat in non_negative_features:
                aug[feat] = np.maximum(aug[feat], 0.0)
        augmented.append(aug)

    return pd.concat(augmented, ignore_index=True)


def augment_triton(df: pd.DataFrame, speed_augments: int = 1,
                   weather_augments: int = 1, seed: int = 42) -> pd.DataFrame:
    """Apply combined physics-based augmentation to Triton data.

    Returns original + augmented data concatenated.
    """
    aug_speed = augment_speed_cubic(df, n_augments=speed_augments, seed=seed)
    aug_weather = augment_weather_noise(df, n_augments=weather_augments, seed=seed + 100)

    combined = pd.concat([df, aug_speed, aug_weather], ignore_index=True)
    return combined
