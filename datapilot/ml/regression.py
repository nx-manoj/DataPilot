import numpy as np
from typing import Union, Dict, Any, List


def regression_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> Dict[str, Any]:
    """Calculates comprehensive regression evaluation metrics.

    Args:
        y_true: Ground-truth continuous target values.
        y_pred: Predicted continuous values from the model.

    Returns:
        Dict containing MAE, MSE, RMSE, R², MAPE, and Max Error.
        Also prints a formatted metrics report to stdout.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mae   = float(np.mean(np.abs(y_t - y_p)))
    mse   = float(np.mean((y_t - y_p) ** 2))
    rmse  = float(np.sqrt(mse))

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # MAPE — guard against zero targets
    nonzero_mask = y_t != 0
    mape = float(np.mean(np.abs((y_t[nonzero_mask] - y_p[nonzero_mask]) / y_t[nonzero_mask])) * 100) \
           if nonzero_mask.any() else float("nan")

    max_err = float(np.max(np.abs(y_t - y_p)))

    print("=" * 50)
    print("     DATAPILOT REGRESSION METRICS REPORT     ")
    print("=" * 50)
    print(f"  MAE        (Mean Absolute Error)   : {mae:.4f}")
    print(f"  MSE        (Mean Squared Error)    : {mse:.4f}")
    print(f"  RMSE       (Root Mean Sq. Error)   : {rmse:.4f}")
    print(f"  R²         (Coefficient of Det.)   : {r2:.4f}")
    print(f"  MAPE       (Mean Abs. % Error)     : {mape:.2f}%")
    print(f"  Max Error  (Worst single pred.)    : {max_err:.4f}")
    print("-" * 50)

    if r2 >= 0.90:
        verdict = "✅ EXCELLENT — model explains ≥90% of variance."
    elif r2 >= 0.75:
        verdict = "🟡 GOOD — solid model, room for improvement."
    elif r2 >= 0.50:
        verdict = "🟠 FAIR — model captures broad trends but misses detail."
    else:
        verdict = "🔴 POOR — model explains <50% of variance. Re-engineer features."

    print(f"  Verdict: {verdict}")
    print("=" * 50)

    return {
        "mae":       round(mae,  4),
        "mse":       round(mse,  4),
        "rmse":      round(rmse, 4),
        "r2":        round(r2,   4),
        "mape":      round(mape, 2) if not np.isnan(mape) else None,
        "max_error": round(max_err, 4),
    }
