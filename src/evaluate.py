"""
evaluate.py
───────────
Evaluation helpers: metrics, McNemar's test, cross-domain degradation table.

Usage
-----
    from src.evaluate import compute_metrics, mcnemar_test, degradation_table
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    """Return accuracy, precision, recall, F1 as a dict."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred),    4),
        "f1":        round(f1_score(y_true, y_pred),        4),
    }


def mcnemar_test(y_true, preds_a, preds_b,
                 name_a: str = "Model A",
                 name_b: str = "Model B",
                 verbose: bool = True) -> tuple[float, float]:
    """
    McNemar's test with continuity correction for paired classifiers.

    Parameters
    ----------
    y_true   : array-like  Ground-truth labels.
    preds_a  : array-like  Predictions from model A.
    preds_b  : array-like  Predictions from model B.
    name_a   : str         Label for model A in printed output.
    name_b   : str         Label for model B in printed output.
    verbose  : bool        Print contingency table and result.

    Returns
    -------
    chi2_stat : float
    p_value   : float
    """
    y_true  = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)

    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    b01 = int(np.sum(~correct_a &  correct_b))   # B right, A wrong
    b10 = int(np.sum( correct_a & ~correct_b))   # A right, B wrong
    b00 = int(np.sum(~correct_a & ~correct_b))
    b11 = int(np.sum( correct_a &  correct_b))

    n = b01 + b10
    if n == 0:
        chi2_stat, p_val = 0.0, 1.0
    else:
        chi2_stat = (abs(b01 - b10) - 1) ** 2 / n
        p_val     = float(1 - chi2_dist.cdf(chi2_stat, df=1))

    if verbose:
        width = max(len(name_a), len(name_b)) + 2
        print(f"McNemar's Test: {name_a} vs {name_b}")
        print(f"  Both correct:                     {b11:>6,}")
        print(f"  Both wrong:                       {b00:>6,}")
        print(f"  {name_a:<{width}} right, {name_b} wrong: {b10:>6,}")
        print(f"  {name_a:<{width}} wrong, {name_b} right: {b01:>6,}")
        print(f"  Discordant pairs:                 {n:>6,}")
        print(f"  chi2 = {chi2_stat:.4f},  p = {p_val:.6f}")
        if p_val < 0.05:
            winner = name_b if b01 > b10 else name_a
            print(f"  Significant (p < 0.05) — {winner} is reliably better")
        else:
            print("  Not significant at p < 0.05")

    return chi2_stat, p_val


def degradation_table(scores_source: dict, scores_target: dict,
                      source_name: str = "Source",
                      target_name: str = "Target") -> pd.DataFrame:
    """
    Build a cross-domain degradation summary from two metric dicts.

    Parameters
    ----------
    scores_source : dict  Metrics on the training domain test set.
    scores_target : dict  Metrics on the out-of-domain test set.

    Returns
    -------
    pd.DataFrame  with columns: metric, source_score, target_score, drop
    """
    rows = []
    for m in ["accuracy", "precision", "recall", "f1"]:
        src = scores_source[m]
        tgt = scores_target[m]
        rows.append({
            "metric":       m.capitalize(),
            source_name:    src,
            target_name:    tgt,
            "drop":         round(src - tgt, 4),
        })
    return pd.DataFrame(rows)
