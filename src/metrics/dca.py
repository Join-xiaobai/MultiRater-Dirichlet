import numpy as np
import pandas as pd

def compute_dca_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    """
    Compute Standardized Net Benefit across clinical referral thresholds p_t.
    """
    n = len(y_true)
    event_rate = np.mean(y_true)
    net_benefits = []
    all_benefits = []
    none_benefits = [0.0] * len(thresholds)

    for pt in thresholds:
        weight = pt / (1.0 - pt)
        y_pred = (y_prob >= pt).astype(float)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * weight
        net_benefits.append(float(nb))

        # Refer All
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all = (tp_all / n) - (fp_all / n) * weight
        all_benefits.append(float(nb_all))

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit": net_benefits,
        "refer_all": all_benefits,
        "refer_none": none_benefits
    }

def simulate_collaborative_triage(uncertainties: np.ndarray, probs: np.ndarray, y_true: np.ndarray, defer_ratio: float = 0.2, pt: float = 0.30) -> dict:
    """
    Simulate human-AI collaborative triage workflow where top X% uncertain cases
    are deferred to multidisciplinary tumor board (MDT).
    """
    n = len(y_true)
    k = int(n * defer_ratio)
    deferred_idx = np.argsort(uncertainties)[-k:]
    autonomous_idx = np.argsort(uncertainties)[:-k]

    weight = pt / (1.0 - pt)
    
    # Autonomous decisions by AI
    auto_pred = (probs[autonomous_idx] >= pt).astype(float)
    tp_auto = np.sum((auto_pred == 1) & (y_true[autonomous_idx] == 1))
    fp_auto = np.sum((auto_pred == 1) & (y_true[autonomous_idx] == 0))

    # Expert panel review on deferred cases (assumed 100% sensitivity, 95% specificity)
    expert_tp = np.sum(y_true[deferred_idx] == 1)
    expert_fp = int(np.sum(y_true[deferred_idx] == 0) * 0.05)

    tp_total = tp_auto + expert_tp
    fp_total = fp_auto + expert_fp

    collaborative_nb = (tp_total / n) - (fp_total / n) * weight
    return {"net_benefit": float(collaborative_nb), "deferred_count": k, "autonomous_count": len(autonomous_idx)}
