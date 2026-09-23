import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def compute_multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Multi-class Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def compute_distribution_metrics(pred_probs: np.ndarray, target_probs: np.ndarray) -> dict:
    """
    Compute Wasserstein distance (Earth Mover Distance) and Mean Squared Error.
    """
    n = len(pred_probs)
    emds = []
    classes = np.arange(1, pred_probs.shape[1] + 1)
    for i in range(n):
        emd = wasserstein_distance(classes, classes, u_weights=pred_probs[i], v_weights=target_probs[i])
        emds.append(emd)
    
    mse = float(np.mean((pred_probs - target_probs) ** 2))
    mean_emd = float(np.mean(emds))
    return {"emd": mean_emd, "mse": mse}

def compute_discrimination_metrics(scores: np.ndarray, conflict_flags: np.ndarray) -> dict:
    """
    Compute AUROC and AUPRC for conflict detection.
    """
    if len(np.unique(conflict_flags)) > 1:
        auroc = float(roc_auc_score(conflict_flags, scores))
        auprc = float(average_precision_score(conflict_flags, scores))
    else:
        auroc, auprc = 0.5, 0.0
    return {"auroc": auroc, "auprc": auprc}

def compute_bootstrap_ci(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Paired bootstrap confidence intervals for AUROC delta.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], y_score_a[idx])
        auc_b = roc_auc_score(y_true[idx], y_score_b[idx])
        diffs.append(auc_b - auc_a)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5))
    }
