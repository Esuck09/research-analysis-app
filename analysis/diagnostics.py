# analysis/diagnostics.py

from __future__ import annotations
import pandas as pd


def _is_loss_metric(name: str) -> bool:
    return "loss" in name.lower()


def diagnose_experiment(exp: dict) -> dict:
    """
    Returns a dict with diagnostics about a single experiment.
    Non-blocking: purely informational.
    """
    df: pd.DataFrame = exp["metrics_df"].copy()

    issues: list[str] = []

    # epoch checks
    if "epoch" not in df.columns:
        issues.append("Missing required column: epoch")
        return {
            "experiment_name": exp.get("experiment_name", ""),
            "file_name": exp.get("source_file", exp.get("file_name", "")),
            "task": exp.get("task", ""),
            "model": exp.get("model", ""),
            "dataset": exp.get("dataset", ""),
            "n_epochs": 0,
            "min_epoch": None,
            "max_epoch": None,
            "duplicate_epochs": True,
            "non_monotonic_epochs": True,
            "missing_epoch_gaps": None,
            "nan_metrics": "N/A",
            "issues": issues,
            "severity": "error",
        }

    # coerce epoch numeric for safe checks
    epoch = pd.to_numeric(df["epoch"], errors="coerce")
    if epoch.isna().all():
        issues.append("Epoch column is non-numeric or empty")
        severity = "error"
        return {
            "experiment_name": exp.get("experiment_name", ""),
            "file_name": exp.get("source_file", exp.get("file_name", "")),
            "task": exp.get("task", ""),
            "model": exp.get("model", ""),
            "dataset": exp.get("dataset", ""),
            "n_epochs": 0,
            "min_epoch": None,
            "max_epoch": None,
            "duplicate_epochs": False,
            "non_monotonic_epochs": True,
            "missing_epoch_gaps": None,
            "nan_metrics": "N/A",
            "issues": issues,
            "severity": severity,
        }

    epoch = epoch.dropna().astype(int)
    n_epochs = int(epoch.nunique())
    min_epoch = int(epoch.min())
    max_epoch = int(epoch.max())

    # duplicate epochs
    duplicate_epochs = df["epoch"].duplicated().any()
    if duplicate_epochs:
        issues.append("Duplicate epochs present (duplicates were likely overwritten/kept-last during validation)")

    # monotonic check
    epoch_sorted = epoch.sort_values()
    non_monotonic = not epoch.is_monotonic_increasing
    if non_monotonic:
        issues.append("Epoch values are not strictly increasing (input order not sorted)")

    # gaps check (only meaningful if epochs look contiguous-ish)
    expected = set(range(min_epoch, max_epoch + 1))
    actual = set(epoch_sorted.tolist())
    missing = sorted(expected - actual)
    missing_epoch_gaps = len(missing)
    if missing_epoch_gaps > 0:
        issues.append(f"Missing {missing_epoch_gaps} epochs between {min_epoch} and {max_epoch}")

    # NaN metrics check
    metric_cols = [c for c in df.columns if c != "epoch"]
    nan_metrics = []
    for c in metric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            nan_metrics.append(f"{c}(all-NaN)")
        elif s.isna().any():
            nan_metrics.append(f"{c}(has-NaN)")

    if nan_metrics:
        issues.append("NaNs detected in metrics: " + ", ".join(nan_metrics))

    # severity
    severity = "ok"
    if any("Missing required column" in x for x in issues) or any("non-numeric" in x for x in issues):
        severity = "error"
    elif issues:
        severity = "warning"

    return {
        "experiment_name": exp.get("experiment_name", ""),
        "file_name": exp.get("source_file", exp.get("file_name", "")),
        "task": exp.get("task", ""),
        "model": exp.get("model", ""),
        "dataset": exp.get("dataset", ""),
        "n_epochs": n_epochs,
        "min_epoch": min_epoch,
        "max_epoch": max_epoch,
        "duplicate_epochs": bool(duplicate_epochs),
        "non_monotonic_epochs": bool(non_monotonic),
        "missing_epoch_gaps": int(missing_epoch_gaps),
        "nan_metrics": ", ".join(nan_metrics) if nan_metrics else "",
        "issues": issues,
        "severity": severity,
    }


def build_diagnostics_table(experiments: list[dict]) -> pd.DataFrame:
    rows = [diagnose_experiment(e) for e in experiments]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # compact issues summary
    df["issues_count"] = df["issues"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["issues_summary"] = df["issues"].apply(
        lambda x: " • ".join(x[:2]) + (" ..." if len(x) > 2 else "") if isinstance(x, list) and x else ""
    )

    # order columns nicely
    col_order = [
        "severity",
        "experiment_name",
        "file_name",
        "task",
        "model",
        "dataset",
        "n_epochs",
        "min_epoch",
        "max_epoch",
        "missing_epoch_gaps",
        "duplicate_epochs",
        "non_monotonic_epochs",
        "nan_metrics",
        "issues_count",
        "issues_summary",
        "issues",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    return df[col_order]


def cross_experiment_checks(experiments: list[dict]) -> list[str]:
    """
    Global warnings across experiments: mismatched epoch ranges and overlaps.
    """
    warnings: list[str] = []
    if not experiments:
        return warnings

    mins = []
    maxs = []
    for e in experiments:
        d = diagnose_experiment(e)
        if d["min_epoch"] is not None and d["max_epoch"] is not None:
            mins.append(d["min_epoch"])
            maxs.append(d["max_epoch"])

    if not mins or not maxs:
        return warnings

    if len(set(mins)) > 1 or len(set(maxs)) > 1:
        warnings.append(
            f"Epoch ranges differ across experiments (min epochs: {sorted(set(mins))}, max epochs: {sorted(set(maxs))}). "
            "Overlays are valid but comparisons at early/late epochs may be unfair."
        )

    common_min = max(mins)
    common_max = min(maxs)
    if common_min > common_max:
        warnings.append("No overlapping epoch range across experiments (cannot compare on a shared epoch window).")
    else:
        warnings.append(f"Common overlapping epoch window: {common_min} → {common_max}")

    return warnings
