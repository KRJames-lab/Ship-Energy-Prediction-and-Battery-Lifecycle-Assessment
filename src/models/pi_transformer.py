"""Physics-Informed Transformer Encoder for marine energy prediction."""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PITransformer(nn.Module):
    """Transformer Encoder for time series regression.

    Architecture:
        Input Projection → Positional Encoding → TransformerEncoder(2 layers, 4 heads)
        → Mean Pooling → Linear → Output
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch,) predictions
        """
        x = self.input_proj(x)                    # (batch, seq_len, d_model)
        x = self.pos_encoder(x)                    # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)            # (batch, seq_len, d_model)
        x = x.mean(dim=1)                          # (batch, d_model) mean pooling
        return self.fc(x).squeeze(-1)              # (batch,)


def train_pi_transformer(
    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    physics_loss_fn, n_features,
    epochs=100, batch_size=256, lr=1e-3, patience=10,
    device="cuda",
    checkpoint_path="checkpoints/poseidon_transformer.pt",
):
    """Train PI-Transformer with physics-informed loss and early stopping.

    Returns:
        (model, history) tuple.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = PITransformer(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_X_t = torch.FloatTensor(X_train_seq)
    train_y_t = torch.FloatTensor(y_train_seq)
    train_flat_t = torch.FloatTensor(X_train_seq[:, -1, :])  # last step

    val_X_t = torch.FloatTensor(X_val_seq)
    val_y_t = torch.FloatTensor(y_val_seq)

    train_ds = TensorDataset(train_X_t, train_y_t, train_flat_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch, X_flat_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_flat_batch = X_flat_batch.to(device).requires_grad_(True)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss, components = physics_loss_fn(y_pred, y_batch, X_flat_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t.to(device))
            val_loss = nn.MSELoss()(val_pred, val_y_t.to(device)).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return model, history


def predict_transformer(model, X_seq, device="cuda"):
    """Generate predictions from trained Transformer."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_seq).to(device)
        return model(X_t).cpu().numpy()
