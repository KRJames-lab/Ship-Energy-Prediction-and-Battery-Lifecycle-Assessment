"""Phase 3 Runner: Triton Fine-tuning with Transfer Learning.

Comparison experiments:
1. XGBoost Scratch vs Transfer (feature reuse)
2. PI-LSTM: Scratch vs Transfer, PI vs Vanilla, Aug vs No-Aug
3. PI-Transformer: Scratch vs Transfer, PI vs Vanilla, Aug vs No-Aug
"""
import sys
sys.path.insert(0, ".")

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from src.data_preprocessing import (
    select_features, time_based_split, create_scaled_datasets,
    create_sequence_dataset, INPUT_FEATURES, TARGET,
)
from src.data_augmentation import augment_triton
from src.physics_loss import PhysicsInformedLoss
from src.models.xgboost_baseline import train_xgboost, evaluate_model
from src.models.pi_lstm import PILSTM, train_pi_lstm, predict_lstm
from src.models.pi_transformer import PITransformer, train_pi_transformer, predict_transformer
from src.evaluate_phase2 import compute_metrics, create_comparison_chart, print_comparison_table

SEQ_LEN = 24
P_HOTEL_TRITON = 1384.0  # Triton hotel load (kW)
LAMBDA_PHYSICS = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FEATURES = len(INPUT_FEATURES)


def finetune_lstm(
    pretrained_path, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    physics_loss_fn, freeze_layers=True,
    epochs=200, batch_size=128, lr=1e-4, patience=20,
    checkpoint_path="checkpoints/triton_lstm_tl.pt",
):
    """Fine-tune pre-trained LSTM on Triton data.

    Args:
        pretrained_path: Path to Poseidon pre-trained weights.
        freeze_layers: If True, freeze LSTM layers and only train FC.
    """
    device = torch.device(DEVICE)

    model = PILSTM(input_size=N_FEATURES).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    if freeze_layers:
        for param in model.lstm.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    else:
        # Fine-tune all with lower LR for pre-trained layers
        optimizer = torch.optim.Adam([
            {"params": model.lstm.parameters(), "lr": lr * 0.1},
            {"params": model.fc.parameters(), "lr": lr},
        ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    from torch.utils.data import DataLoader, TensorDataset

    train_X_t = torch.FloatTensor(X_train_seq)
    train_y_t = torch.FloatTensor(y_train_seq)
    train_flat_t = torch.FloatTensor(X_train_seq[:, -1, :])

    val_X_t = torch.FloatTensor(X_val_seq).to(device)
    val_y_t = torch.FloatTensor(y_val_seq).to(device)

    train_ds = TensorDataset(train_X_t, train_y_t, train_flat_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch, X_flat in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_flat = X_flat.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss, _ = physics_loss_fn(y_pred, y_batch, X_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = nn.MSELoss()(val_pred, val_y_t).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Train: {epoch_loss/n_batches:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return model


def finetune_transformer(
    pretrained_path, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
    physics_loss_fn, freeze_layers=True,
    epochs=200, batch_size=128, lr=1e-4, patience=20,
    checkpoint_path="checkpoints/triton_transformer_tl.pt",
):
    """Fine-tune pre-trained Transformer on Triton data."""
    device = torch.device(DEVICE)

    model = PITransformer(input_size=N_FEATURES).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    if freeze_layers:
        for param in model.input_proj.parameters():
            param.requires_grad = False
        for param in model.pos_encoder.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {"params": list(model.input_proj.parameters()) +
                       list(model.transformer_encoder.parameters()), "lr": lr * 0.1},
            {"params": model.fc.parameters(), "lr": lr},
        ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    from torch.utils.data import DataLoader, TensorDataset

    train_X_t = torch.FloatTensor(X_train_seq)
    train_y_t = torch.FloatTensor(y_train_seq)
    train_flat_t = torch.FloatTensor(X_train_seq[:, -1, :])

    val_X_t = torch.FloatTensor(X_val_seq).to(device)
    val_y_t = torch.FloatTensor(y_val_seq).to(device)

    train_ds = TensorDataset(train_X_t, train_y_t, train_flat_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch, X_flat in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_flat = X_flat.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss, _ = physics_loss_fn(y_pred, y_batch, X_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = nn.MSELoss()(val_pred, val_y_t).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Train: {epoch_loss/n_batches:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return model


def prepare_triton_data(augment=False):
    """Load and preprocess Triton data, optionally with augmentation."""
    df = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
    df_feat = select_features(df)
    train, val, test = time_based_split(df_feat)

    if augment:
        # Augment training data only (before scaling)
        train_raw = pd.read_parquet("Dataset/CPS_Triton_with_energy.parquet")
        train_raw_feat = select_features(train_raw)
        train_raw_split = train_raw_feat.iloc[:len(train)]
        train_augmented = augment_triton(train_raw_split, speed_augments=1, weather_augments=1)
        train = train_augmented.reset_index(drop=True)

    datasets, scaler_X, scaler_y = create_scaled_datasets(train, val, test)
    return datasets, scaler_X, scaler_y, test


def run_experiment(name, datasets, scaler_y, model_type, transfer=False,
                   physics=True, augmented=False):
    """Run a single experiment configuration."""
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    p_hotel_scaled = (P_HOTEL_TRITON - scaler_y.mean_[0]) / scaler_y.scale_[0]

    if physics:
        loss_fn = PhysicsInformedLoss(p_hotel=p_hotel_scaled, lambda_physics=LAMBDA_PHYSICS)
    else:
        loss_fn = PhysicsInformedLoss(p_hotel=p_hotel_scaled, lambda_physics=0.0)

    if model_type == "xgboost":
        model = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_eval = evaluate_model(model, X_test, y_test, scaler_y)
        return {
            "metrics": {k: xgb_eval[k] for k in ["RMSE", "MAE", "R2", "MAPE"]},
            "y_pred": xgb_eval["y_pred"],
            "y_test": xgb_eval["y_test"],
        }

    # Sequence data
    X_train_seq, y_train_seq = create_sequence_dataset(X_train, y_train, SEQ_LEN)
    X_val_seq, y_val_seq = create_sequence_dataset(X_val, y_val, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequence_dataset(X_test, y_test, SEQ_LEN)

    if model_type == "lstm":
        if transfer:
            model = finetune_lstm(
                "checkpoints/poseidon_lstm.pt",
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                physics_loss_fn=loss_fn, freeze_layers=False,
                checkpoint_path=f"checkpoints/triton_lstm_{name}.pt",
            )
        else:
            from src.models.pi_lstm import train_pi_lstm
            model, _ = train_pi_lstm(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                physics_loss_fn=loss_fn, n_features=N_FEATURES,
                epochs=200, batch_size=128, lr=5e-4, patience=20,
                device=DEVICE,
                checkpoint_path=f"checkpoints/triton_lstm_{name}.pt",
            )
        y_pred_scaled = predict_lstm(model, X_test_seq, device=DEVICE)

    elif model_type == "transformer":
        if transfer:
            model = finetune_transformer(
                "checkpoints/poseidon_transformer.pt",
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                physics_loss_fn=loss_fn, freeze_layers=False,
                checkpoint_path=f"checkpoints/triton_tf_{name}.pt",
            )
        else:
            from src.models.pi_transformer import train_pi_transformer
            model, _ = train_pi_transformer(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                physics_loss_fn=loss_fn, n_features=N_FEATURES,
                epochs=200, batch_size=128, lr=5e-4, patience=20,
                device=DEVICE,
                checkpoint_path=f"checkpoints/triton_tf_{name}.pt",
            )
        y_pred_scaled = predict_transformer(model, X_test_seq, device=DEVICE)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    metrics = compute_metrics(y_true, y_pred)

    return {"metrics": metrics, "y_pred": y_pred, "y_test": y_true}


def main():
    print(f"Phase 3: Triton Fine-tuning")
    print(f"Device: {DEVICE}")
    print(f"P_hotel (Triton): {P_HOTEL_TRITON} kW\n")

    results = {}

    # ===========================
    # Prepare data (no augmentation)
    # ===========================
    print("[1/8] Preparing Triton data (no augmentation)...")
    datasets_noaug, scaler_X_noaug, scaler_y_noaug, test_noaug = prepare_triton_data(augment=False)
    X_train, _ = datasets_noaug["train"]
    print(f"  Train: {X_train.shape[0]}, Val: {datasets_noaug['val'][0].shape[0]}, Test: {datasets_noaug['test'][0].shape[0]}")

    # ===========================
    # Prepare data (with augmentation)
    # ===========================
    print("\n[2/8] Preparing Triton data (with augmentation)...")
    datasets_aug, scaler_X_aug, scaler_y_aug, test_aug = prepare_triton_data(augment=True)
    X_train_aug, _ = datasets_aug["train"]
    print(f"  Train (augmented): {X_train_aug.shape[0]}")

    # ===========================
    # Experiment 1: XGBoost Scratch
    # ===========================
    print("\n[3/8] XGBoost Scratch...")
    results["XGB-Scratch"] = run_experiment(
        "xgb_scratch", datasets_noaug, scaler_y_noaug, "xgboost")
    print(f"  R²={results['XGB-Scratch']['metrics']['R2']:.4f}")

    # ===========================
    # Experiment 2: LSTM Scratch (PI)
    # ===========================
    print("\n[4/8] PI-LSTM Scratch...")
    results["LSTM-Scratch-PI"] = run_experiment(
        "lstm_scratch_pi", datasets_noaug, scaler_y_noaug, "lstm",
        transfer=False, physics=True)
    print(f"  R²={results['LSTM-Scratch-PI']['metrics']['R2']:.4f}")

    # ===========================
    # Experiment 3: LSTM Transfer (PI, full fine-tune)
    # ===========================
    print("\n[5/8] PI-LSTM Transfer (full fine-tune)...")
    results["LSTM-TL-PI"] = run_experiment(
        "lstm_tl_pi", datasets_noaug, scaler_y_noaug, "lstm",
        transfer=True, physics=True)
    print(f"  R²={results['LSTM-TL-PI']['metrics']['R2']:.4f}")

    # ===========================
    # Experiment 4: LSTM Transfer (Vanilla, no physics)
    # ===========================
    print("\n[6/8] LSTM Transfer (Vanilla, no physics)...")
    results["LSTM-TL-Vanilla"] = run_experiment(
        "lstm_tl_vanilla", datasets_noaug, scaler_y_noaug, "lstm",
        transfer=True, physics=False)
    print(f"  R²={results['LSTM-TL-Vanilla']['metrics']['R2']:.4f}")

    # ===========================
    # Experiment 5: LSTM Transfer + PI + Augmentation
    # ===========================
    print("\n[7/8] PI-LSTM Transfer + Augmentation...")
    results["LSTM-TL-PI-Aug"] = run_experiment(
        "lstm_tl_pi_aug", datasets_aug, scaler_y_aug, "lstm",
        transfer=True, physics=True, augmented=True)
    print(f"  R²={results['LSTM-TL-PI-Aug']['metrics']['R2']:.4f}")

    # ===========================
    # Experiment 6: Transformer Transfer (PI, full fine-tune)
    # ===========================
    print("\n[8/8] PI-Transformer Transfer (full fine-tune)...")
    results["TF-TL-PI"] = run_experiment(
        "tf_tl_pi", datasets_noaug, scaler_y_noaug, "transformer",
        transfer=True, physics=True)
    print(f"  R²={results['TF-TL-PI']['metrics']['R2']:.4f}")

    # ===========================
    # Results
    # ===========================
    print_comparison_table(results)

    # Ablation summary
    print("\n--- Ablation Analysis ---")
    lstm_scratch = results["LSTM-Scratch-PI"]["metrics"]["R2"]
    lstm_tl = results["LSTM-TL-PI"]["metrics"]["R2"]
    lstm_vanilla = results["LSTM-TL-Vanilla"]["metrics"]["R2"]
    lstm_aug = results["LSTM-TL-PI-Aug"]["metrics"]["R2"]

    print(f"  Transfer Learning effect:  R² {lstm_scratch:.4f} → {lstm_tl:.4f} (Δ={lstm_tl-lstm_scratch:+.4f})")
    print(f"  Physics Loss effect:       R² {lstm_vanilla:.4f} → {lstm_tl:.4f} (Δ={lstm_tl-lstm_vanilla:+.4f})")
    print(f"  Augmentation effect:       R² {lstm_tl:.4f} → {lstm_aug:.4f} (Δ={lstm_aug-lstm_tl:+.4f})")

    # Save chart with key experiments
    key_results = {k: results[k] for k in [
        "XGB-Scratch", "LSTM-Scratch-PI", "LSTM-TL-PI", "LSTM-TL-PI-Aug", "TF-TL-PI"
    ]}
    create_comparison_chart(key_results, save_path="reports/phase3_comparison.png")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
