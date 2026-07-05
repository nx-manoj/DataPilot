from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Union, Dict, Any, List

def classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = "auto"
) -> Dict[str, Any]:
    """Calculates key classification performance metrics safely and structured.

    Supports both binary and multi-class targets. When average='auto', the
    function detects the number of unique classes and chooses 'binary' or
    'weighted' automatically.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels from the model.
        average: Averaging strategy for multi-class metrics. One of 'auto',
                 'binary', 'weighted', 'macro', or 'micro'. Defaults to 'auto'.

    Returns:
        dict: Keys 'accuracy', 'precision', 'recall', 'f1_score'.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Auto-detect averaging strategy based on unique class count
    if average == "auto":
        n_classes = len(np.unique(y_t))
        avg = "binary" if n_classes == 2 else "weighted"
    else:
        avg = average

    return {
        "accuracy": round(accuracy_score(y_t, y_p), 4),
        "precision": round(precision_score(y_t, y_p, average=avg, zero_division=0), 4),
        "recall": round(recall_score(y_t, y_p, average=avg, zero_division=0), 4),
        "f1_score": round(f1_score(y_t, y_p, average=avg, zero_division=0), 4),
    }
