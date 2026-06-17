"""Physics-Informed LSTM for marine energy prediction."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PILSTM(nn.Module):
    """3-layer LSTM with linear output head.

    Architecture:
        Input → LSTM(3 layers, hidden=256) → Dropout → Linear → Output
    """

    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch,) predictions
        """
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden)
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)      # (batch,)


def train_pi_lstm(
    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    physics_loss_fn, n_features,
    X_train_flat_last=None,  # last step features for physics loss
    X_val_flat_last=None,
    epochs=100, batch_size=256, lr=1e-3, patience=10,
    device="cuda",
    checkpoint_path="checkpoints/poseidon_lstm.pt",
):
    """Train PI-LSTM with physics-informed loss and early stopping.

    Args:
        X_train_seq: (n_train, seq_len, n_features) numpy array.
        y_train_seq: (n_train,) numpy array.
        X_val_seq: (n_val, seq_len, n_features) numpy array.
        y_val_seq: (n_val,) numpy array.
        physics_loss_fn: PhysicsInformedLoss instance.
        n_features: Number of input features.
        X_train_flat_last: (n_train, n_features) last time step for physics grad.
        X_val_flat_last: (n_val, n_features) last time step for physics grad.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        device: torch device.
        checkpoint_path: Path to save best model.

    Returns:
        (model, history) tuple.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = PILSTM(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Prepare data loaders
    train_X_t = torch.FloatTensor(X_train_seq)
    train_y_t = torch.FloatTensor(y_train_seq)
    train_flat_t = torch.FloatTensor(
        X_train_flat_last if X_train_flat_last is not None else X_train_seq[:, -1, :]
    )

    val_X_t = torch.FloatTensor(X_val_seq)
    val_y_t = torch.FloatTensor(y_val_seq)
    val_flat_t = torch.FloatTensor(
        X_val_flat_last if X_val_flat_last is not None else X_val_seq[:, -1, :]
    )

    train_ds = TensorDataset(train_X_t, train_y_t, train_flat_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "components": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_X_dev = val_X_t.to(device)
            val_y_dev = val_y_t.to(device)
            val_pred = model(val_X_dev)
            val_loss = nn.MSELoss()(val_pred, val_y_dev).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history


def predict_lstm(model, X_seq, device="cuda"):
    """Generate predictions from trained LSTM."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_seq).to(device)
        return model(X_t).cpu().numpy()
