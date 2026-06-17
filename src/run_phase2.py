"""Phase 2 Runner: Poseidon Pre-training with XGBoost, PI-LSTM, PI-Transformer."""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
import pandas as pd
import torch
import joblib

from src.data_preprocessing import (
    select_features, time_based_split, create_scaled_datasets,
    create_sequence_dataset, INPUT_FEATURES, TARGET,
)
from src.physics_loss import PhysicsInformedLoss
from src.models.xgboost_baseline import train_xgboost, evaluate_model
from src.models.pi_lstm import PILSTM, train_pi_lstm, predict_lstm
from src.models.pi_transformer import PITransformer, train_pi_transformer, predict_transformer
from src.evaluate_phase2 import (
    compute_metrics, compute_physics_violations,
    create_comparison_chart, print_comparison_table,
)

SEQ_LEN = 24
P_HOTEL = 7695.0  # Poseidon hotel load (kW, median at speed < 0.5 knots)
LAMBDA_PHYSICS = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    print(f"Physics: lambda={LAMBDA_PHYSICS}, P_hotel={P_HOTEL}")

    # --- Load & preprocess ---
    print("\n[1/6] Loading Poseidon data...")
    df = pd.read_parquet("Dataset/CPS_Poseidon_with_energy.parquet")
    df_feat = select_features(df)
    print(f"  Features: {len(INPUT_FEATURES)} inputs, shape: {df_feat.shape}")

    print("\n[2/6] Splitting & scaling...")
    train, val, test = time_based_split(df_feat)
    datasets, scaler_X, scaler_y = create_scaled_datasets(train, val, test)
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Save scalers for Phase 3
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(scaler_X, "checkpoints/poseidon_scaler_X.pkl")
    joblib.dump(scaler_y, "checkpoints/poseidon_scaler_y.pkl")

    # Sequence datasets for LSTM/Transformer
    X_train_seq, y_train_seq = create_sequence_dataset(X_train, y_train, SEQ_LEN)
    X_val_seq, y_val_seq = create_sequence_dataset(X_val, y_val, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequence_dataset(X_test, y_test, SEQ_LEN)
    print(f"  Sequences: train={X_train_seq.shape[0]}, val={X_val_seq.shape[0]}, test={X_test_seq.shape[0]}")

    results = {}

    # ===========================================
    # Model 1: XGBoost Baseline (no physics loss)
    # ===========================================
    print("\n[3/6] Training XGBoost baseline...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_eval = evaluate_model(xgb_model, X_test, y_test, scaler_y)
    joblib.dump(xgb_model, "checkpoints/poseidon_xgboost.pkl")

    results["XGBoost (Baseline)"] = {
        "metrics": {k: xgb_eval[k] for k in ["RMSE", "MAE", "R2", "MAPE"]},
        "y_pred": xgb_eval["y_pred"],
        "y_test": xgb_eval["y_test"],
    }
    print(f"  XGBoost: R²={xgb_eval['R2']:.4f}, RMSE={xgb_eval['RMSE']:.1f}")

    # ===========================================
    # Model 2: PI-LSTM
    # ===========================================
    print("\n[4/6] Training PI-LSTM...")
    # Scale P_hotel to match scaler_y
    p_hotel_scaled = (P_HOTEL - scaler_y.mean_[0]) / scaler_y.scale_[0]
    physics_loss = PhysicsInformedLoss(p_hotel=p_hotel_scaled, lambda_physics=LAMBDA_PHYSICS)

    lstm_model, lstm_history = train_pi_lstm(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        physics_loss_fn=physics_loss,
        n_features=len(INPUT_FEATURES),
        epochs=200, batch_size=128, lr=5e-4, patience=20,
        device=DEVICE,
        checkpoint_path="checkpoints/poseidon_lstm.pt",
    )

    lstm_pred_scaled = predict_lstm(lstm_model, X_test_seq, device=DEVICE)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
    lstm_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    lstm_metrics = compute_metrics(lstm_true, lstm_pred)

    results["PI-LSTM"] = {
        "metrics": lstm_metrics,
        "y_pred": lstm_pred,
        "y_test": lstm_true,
    }
    print(f"  PI-LSTM: R²={lstm_metrics['R2']:.4f}, RMSE={lstm_metrics['RMSE']:.1f}")

    # ===========================================
    # Model 3: PI-Transformer
    # ===========================================
    print("\n[5/6] Training PI-Transformer...")
    physics_loss_tf = PhysicsInformedLoss(p_hotel=p_hotel_scaled, lambda_physics=LAMBDA_PHYSICS)

    tf_model, tf_history = train_pi_transformer(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        physics_loss_fn=physics_loss_tf,
        n_features=len(INPUT_FEATURES),
        epochs=200, batch_size=128, lr=5e-4, patience=20,
        device=DEVICE,
        checkpoint_path="checkpoints/poseidon_transformer.pt",
    )

    tf_pred_scaled = predict_transformer(tf_model, X_test_seq, device=DEVICE)
    tf_pred = scaler_y.inverse_transform(tf_pred_scaled.reshape(-1, 1)).ravel()
    tf_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    tf_metrics = compute_metrics(tf_true, tf_pred)

    results["PI-Transformer"] = {
        "metrics": tf_metrics,
        "y_pred": tf_pred,
        "y_test": tf_true,
    }
    print(f"  PI-Transformer: R²={tf_metrics['R2']:.4f}, RMSE={tf_metrics['RMSE']:.1f}")

    # ===========================================
    # Comparison
    # ===========================================
    print("\n[6/6] Generating comparison...")
    print_comparison_table(results)

    # Physics violation analysis
    speed_test = test[INPUT_FEATURES].values[:len(lstm_true), INPUT_FEATURES.index("Ship_SpeedOverGround")]
    wave_test = test[INPUT_FEATURES].values[:len(lstm_true), INPUT_FEATURES.index("Weather_WaveHeight")]

    print(f"\n--- Physics Constraint Violations (Test Set) ---")
    for name, res in results.items():
        y_p = res["y_pred"][:len(speed_test)]
        violations = compute_physics_violations(y_p, speed_test, wave_test, P_HOTEL)
        print(f"  {name}: hotel_violation={violations['hotel_violation_pct']:.1f}%, "
              f"cube_corr={violations['cube_correlation']:.4f}")

    create_comparison_chart(results)

    print("\nPhase 2 complete. Checkpoints saved in checkpoints/")


if __name__ == "__main__":
    main()
