"""Tests for Physics-Informed Loss module (gradient-free formulation)."""
import numpy as np
import torch
import pytest

from src.physics_loss import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    return PhysicsInformedLoss(p_hotel=7695.0, lambda_physics=0.1)


@pytest.fixture
def sample_data():
    """Create sample tensors."""
    torch.manual_seed(42)
    batch = 32
    X = torch.randn(batch, 28)
    y_pred = torch.randn(batch, requires_grad=True) * 5000 + 15000
    y_true = torch.randn(batch) * 5000 + 15000
    return X, y_pred, y_true


class TestPhysicsLossCreation:
    def test_default_params(self, loss_fn):
        assert loss_fn.p_hotel == 7695.0
        assert loss_fn.lambda_physics == 0.1

    def test_custom_lambda(self):
        fn = PhysicsInformedLoss(p_hotel=1000, lambda_physics=0.5)
        assert fn.lambda_physics == 0.5


class TestHotelLoss:
    def test_no_violation(self, loss_fn):
        y_pred = torch.tensor([8000.0, 10000.0, 20000.0])
        loss = loss_fn._hotel_loss(y_pred, p_hotel=7695.0)
        assert loss.item() == 0.0

    def test_violation_penalized(self, loss_fn):
        y_pred = torch.tensor([5000.0, 3000.0])
        loss = loss_fn._hotel_loss(y_pred, p_hotel=7695.0)
        assert loss.item() > 0.0

    def test_mixed(self, loss_fn):
        y_pred = torch.tensor([8000.0, 5000.0])
        loss = loss_fn._hotel_loss(y_pred, p_hotel=7695.0)
        assert loss.item() > 0.0


class TestCubeLoss:
    def test_non_negative(self, loss_fn, sample_data):
        X, y_pred, _ = sample_data
        loss = loss_fn._cube_loss(y_pred, X)
        assert loss.item() >= 0.0

    def test_perfect_cubic_low_loss(self, loss_fn):
        """When P = k * V³ exactly, cube loss should be near zero."""
        X = torch.zeros(50, 28)
        speeds = torch.linspace(0.1, 3.0, 50)
        X[:, 26] = speeds
        k = 100.0
        y_pred = k * speeds ** 3
        loss = loss_fn._cube_loss(y_pred, X)
        assert loss.item() < 0.01


class TestMonotonicityLoss:
    def test_wind_loss_non_negative(self, loss_fn, sample_data):
        X, y_pred, _ = sample_data
        loss = loss_fn._wind_loss(y_pred, X)
        assert loss.item() >= 0.0

    def test_wave_loss_non_negative(self, loss_fn, sample_data):
        X, y_pred, _ = sample_data
        loss = loss_fn._wave_loss(y_pred, X)
        assert loss.item() >= 0.0

    def test_perfectly_monotonic_low_loss(self, loss_fn):
        """When P increases with wind, loss should be low."""
        X = torch.zeros(100, 28)
        wind = torch.linspace(0, 10, 100)
        X[:, 21] = wind
        y_pred = wind * 1000 + 5000  # perfectly correlated
        loss = loss_fn._wind_loss(y_pred, X)
        assert loss.item() == 0.0


class TestTotalLoss:
    def test_total_greater_than_data_loss(self, loss_fn, sample_data):
        X, y_pred, y_true = sample_data
        total, components = loss_fn(y_pred, y_true, X)
        assert "data" in components
        assert "physics" in components
        assert total.item() >= components["data"]

    def test_zero_lambda_equals_data_only(self, sample_data):
        fn = PhysicsInformedLoss(p_hotel=7695.0, lambda_physics=0.0)
        X, y_pred, y_true = sample_data
        total, components = fn(y_pred, y_true, X)
        assert abs(total.item() - components["data"]) < 1e-6

    def test_backward_pass(self, loss_fn):
        torch.manual_seed(42)
        X = torch.randn(32, 28)
        y_pred = torch.randn(32, requires_grad=True) * 5000 + 15000
        y_pred.retain_grad()
        y_true = torch.randn(32) * 5000 + 15000
        total, _ = loss_fn(y_pred, y_true, X)
        total.backward()
        assert y_pred.grad is not None
