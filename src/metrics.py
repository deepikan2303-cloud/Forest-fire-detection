"""Binary classification metrics (fire = positive class) implemented with NumPy only."""
import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool)
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(scores) >= threshold
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "auc": roc_auc(y_true, scores),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
    }


def best_threshold(y_true, scores, target_recall: float | None = None) -> float:
    """Max-F1 threshold, or the highest threshold that still achieves ``target_recall``.

    For early-warning systems a missed fire costs far more than a false alarm,
    so a recall target (e.g. 0.97) is usually the better operating point.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    results = [binary_metrics(y_true, scores, t) for t in grid]
    if target_recall is not None:
        ok = [m for m in results if m["recall"] >= target_recall]
        if ok:
            return float(max(ok, key=lambda m: m["threshold"])["threshold"])
    return float(max(results, key=lambda m: (m["f1"], -abs(m["threshold"] - 0.5)))["threshold"])


def format_metrics(m: dict) -> str:
    return (f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"f1={m['f1']:.4f} auc={m['auc']:.4f}")
