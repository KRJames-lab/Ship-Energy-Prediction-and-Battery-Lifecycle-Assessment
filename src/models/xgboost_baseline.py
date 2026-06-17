"""XGBoost baseline model for energy prediction (no physics loss)."""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost regressor with early stopping.

    Returns:
        Trained XGBRegressor model.
    """
    default_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def evaluate_model(model, X_test, y_test, scaler_y=None):
    """Evaluate model and return metrics in original scale.

    Args:
        model: Trained model with .predict() method.
        X_test: Test features (scaled).
        y_test: Test targets (scaled).
        scaler_y: If provided, inverse-transform predictions and targets.

    Returns:
        dict with RMSE, MAE, R2, MAPE metrics.
    """
    y_pred = model.predict(X_test)

    if scaler_y is not None:
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_pred_orig = y_pred
        y_test_orig = y_test

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    # MAPE (avoid division by zero)
    mask = np.abs(y_test_orig) > 1e-6
    mape = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "y_pred": y_pred_orig,
        "y_test": y_test_orig,
    }
