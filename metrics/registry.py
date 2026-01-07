"""
Metric registry and optimisation rules.
"""

CLASSIFICATION_METRICS = {
    "accuracy": {"optimize": "max"},
    "precision": {"optimize": "max"},
    "recall": {"optimize": "max"},
    "f1": {"optimize": "max"},
    "roc_auc": {"optimize": "max"},
}

SEGMENTATION_METRICS = {
    "dice": {"optimize": "max"},
    "iou": {"optimize": "max"},
    "sensitivity": {"optimize": "max"},
    "specificity": {"optimize": "max"},
    "loss": {"optimize": "min"},
}


def is_classification_metric(metric: str) -> bool:
    return metric in CLASSIFICATION_METRICS


def is_segmentation_metric(metric: str) -> bool:
    return metric in SEGMENTATION_METRICS


def optimization_direction(metric: str) -> str:
    """
    Returns 'max' or 'min' depending on metric.
    Defaults to 'max' if unknown.
    """
    if metric in CLASSIFICATION_METRICS:
        return CLASSIFICATION_METRICS[metric]["optimize"]
    if metric in SEGMENTATION_METRICS:
        return SEGMENTATION_METRICS[metric]["optimize"]
    return "max"
