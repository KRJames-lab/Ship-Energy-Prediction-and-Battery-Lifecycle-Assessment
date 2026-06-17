"""Physics-Informed Loss for marine energy prediction.

Constraints (gradient-free formulation for compatibility with seq2one models):
    L_cube:  P ∝ V³ → penalize deviation from cubic relationship
    L_wind:  Monotonic increase of P w.r.t. wind speed (pair-wise)
    L_wave:  Monotonic increase of P w.r.t. wave height (pair-wise)
    L_hotel: P ≥ P_hotel (minimum hotel load)
"""
import torch
import torch.nn as nn

# Feature indices in INPUT_FEATURES (after preprocessing)
# WEATHER_FEATURES[0..25] + Ship_SpeedOverGround[26] + operating_state[27]
IDX_SPEED = 26
IDX_WIND_SPEED = 21    # Weather_WindSpeed10M
IDX_WAVE_HEIGHT = 16   # Weather_WaveHeight


class PhysicsInformedLoss(nn.Module):
    """Combined data + physics loss for marine energy prediction.

    L_total = L_data + lambda * (L_cube + L_wind + L_wave + L_hotel)

    Uses gradient-free physics penalties based on input-output relationships,
    compatible with any model architecture (XGBoost, LSTM, Transformer).
    """

    def __init__(self, p_hotel: float = 7695.0, lambda_physics: float = 0.1):
        super().__init__()
        self.p_hotel = p_hotel
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def _hotel_loss(self, y_pred: torch.Tensor, p_hotel: float) -> torch.Tensor:
        """Penalize predictions below minimum hotel load.

        L_hotel = mean(ReLU(P_hotel - P_pred)²)
        """
        violation = torch.relu(p_hotel - y_pred)
        return (violation ** 2).mean()

    def _cube_loss(self, y_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Enforce P ∝ V³ by penalizing residuals from cubic fit.

        For cruising samples (speed > 0.5), fit P = k * V³ and penalize
        deviations: L = mean((P_pred - k * V³)²) / var(P_pred)
        """
        speed = X[:, IDX_SPEED]

        # Only apply to moving samples
        mask = speed > 0.0  # in scaled space, ~0 means low speed
        if mask.sum() < 10:
            return torch.tensor(0.0, device=y_pred.device)

        v3 = speed[mask] ** 3
        p = y_pred[mask]

        # Estimate k via least squares: k = sum(P * V³) / sum(V³ * V³)
        k = (p * v3).sum() / (v3 * v3 + 1e-8).sum()
        residual = p - k * v3

        # Normalize by prediction variance to keep scale-invariant
        return (residual ** 2).mean() / (p.var() + 1e-8)

    def _wind_loss(self, y_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Enforce monotonic increase of P w.r.t. wind speed.

        Sample random pairs; if wind_i > wind_j, then P_i should >= P_j.
        L = mean(ReLU(P_j - P_i)²) for pairs where wind_i > wind_j.
        """
        return self._monotonicity_loss(y_pred, X[:, IDX_WIND_SPEED])

    def _wave_loss(self, y_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Enforce monotonic increase of P w.r.t. wave height.

        Same pair-wise approach as wind loss.
        """
        return self._monotonicity_loss(y_pred, X[:, IDX_WAVE_HEIGHT])

    def _monotonicity_loss(
        self, y_pred: torch.Tensor, feature: torch.Tensor, n_pairs: int = 256
    ) -> torch.Tensor:
        """Pair-wise monotonicity penalty.

        Randomly sample pairs (i, j); if feature_i > feature_j,
        penalize cases where y_pred_i < y_pred_j.
        """
        n = len(y_pred)
        if n < 2:
            return torch.tensor(0.0, device=y_pred.device)

        # Random pair indices
        n_pairs = min(n_pairs, n * (n - 1) // 2)
        idx_i = torch.randint(0, n, (n_pairs,), device=y_pred.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=y_pred.device)

        feat_diff = feature[idx_i] - feature[idx_j]  # positive if i has higher feature
        pred_diff = y_pred[idx_i] - y_pred[idx_j]    # should also be positive

        # Only penalize when feature_i > feature_j but pred_i < pred_j
        should_be_higher = feat_diff > 0
        violation = torch.relu(-pred_diff[should_be_higher])

        if len(violation) == 0:
            return torch.tensor(0.0, device=y_pred.device)

        return (violation ** 2).mean()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, X: torch.Tensor
    ) -> tuple:
        """Compute total loss with physics constraints.

        Args:
            y_pred: Predicted power (batch,).
            y_true: True power (batch,).
            X: Input features at last time step (batch, n_features).

        Returns:
            (total_loss, components_dict)
        """
        l_data = self.mse(y_pred, y_true)

        if self.lambda_physics == 0.0:
            return l_data, {
                "data": l_data.item(),
                "physics": 0.0,
                "cube": 0.0,
                "wind": 0.0,
                "wave": 0.0,
                "hotel": 0.0,
            }

        l_cube = self._cube_loss(y_pred, X)
        l_wind = self._wind_loss(y_pred, X)
        l_wave = self._wave_loss(y_pred, X)
        l_hotel = self._hotel_loss(y_pred, self.p_hotel)

        l_physics = l_cube + l_wind + l_wave + l_hotel
        total = l_data + self.lambda_physics * l_physics

        return total, {
            "data": l_data.item(),
            "physics": l_physics.item(),
            "cube": l_cube.item(),
            "wind": l_wind.item(),
            "wave": l_wave.item(),
            "hotel": l_hotel.item(),
        }
