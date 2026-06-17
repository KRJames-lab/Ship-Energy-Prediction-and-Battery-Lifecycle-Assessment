"""Tests for model architectures."""
import numpy as np
import torch
import pytest

from src.models.pi_lstm import PILSTM
from src.models.pi_transformer import PITransformer, PositionalEncoding


N_FEATURES = 28
SEQ_LEN = 24
BATCH = 16


class TestPILSTM:
    def test_output_shape(self):
        model = PILSTM(input_size=N_FEATURES)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (BATCH,)

    def test_single_sample(self):
        model = PILSTM(input_size=N_FEATURES)
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_gradient_flow(self):
        model = PILSTM(input_size=N_FEATURES)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self):
        model = PILSTM(input_size=N_FEATURES)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 5_000_000  # reasonable size


class TestPITransformer:
    def test_output_shape(self):
        model = PITransformer(input_size=N_FEATURES)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (BATCH,)

    def test_single_sample(self):
        model = PITransformer(input_size=N_FEATURES)
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_gradient_flow(self):
        model = PITransformer(input_size=N_FEATURES)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(BATCH, SEQ_LEN, 64)
        out = pe(x)
        assert out.shape == x.shape
