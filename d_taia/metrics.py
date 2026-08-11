from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    labels = list(range(num_classes))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def confidence_interval_95(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    sem = np.std(values, ddof=1) / np.sqrt(n)
    return float(1.96 * sem)


def summarize_runs(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "ci95": confidence_interval_95(arr), "n": len(arr)}