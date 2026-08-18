import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:,1])
        except Exception:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    return metrics
