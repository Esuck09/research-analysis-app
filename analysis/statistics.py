# analysis/statistics.py

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _is_loss_metric(metric: str) -> bool:
    return "loss" in metric.lower()


def _metric_value_at(exp: dict, metric: str, mode: str) -> float | None:
    """
    mode:
      - "final": value at final epoch
      - "best": best value (min for loss, max otherwise)
      - "auc": area under curve over common epochs (simple trapezoid)
    """
    df = exp["metrics_df"]
    if metric not in df.columns:
        return None

    s = df[["epoch", metric]].dropna()
    if s.empty:
        return None

    minimize = _is_loss_metric(metric)

    if mode == "final":
        return float(s[metric].iloc[-1])

    if mode == "best":
        return float(s[metric].min() if minimize else s[metric].max())

    if mode == "auc":
        # AUC over logged epochs (no interpolation)
        x = s["epoch"].to_numpy(dtype=float)
        y = s[metric].to_numpy(dtype=float)
        if len(x) < 2:
            return None
        return float(np.trapz(y, x))

    raise ValueError(f"Unknown mode: {mode}")


def _cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for independent samples.
    """
    a = a.astype(float)
    b = b.astype(float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def _cohens_dz_paired(diffs: np.ndarray) -> float:
    """
    Cohen's dz for paired samples: mean(diff) / std(diff).
    """
    diffs = diffs.astype(float)
    if len(diffs) < 2:
        return np.nan
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diffs) / sd)


def compute_pairwise_stats(
    experiments: list[dict],
    metric: str,
    group_field: str = "model",
    value_mode: str = "final",
    pairing_field: str | None = None,
) -> pd.DataFrame:
    """
    Pairwise statistical comparison across groups.

    Parameters
    ----------
    experiments : list[dict]
    metric : str
    group_field : str
        Which metadata field defines groups to compare (e.g. "model", "group")
    value_mode : str
        "final" | "best" | "auc"
    pairing_field : str | None
        If provided, enforce pairing by this key (e.g. "group" or a custom tag like "seed").
        Pairing means we compare matched items across groups.
        If pairing_field is None, comparisons are unpaired.

    Returns
    -------
    DataFrame with:
      group_a, group_b, n_a, n_b, test_used, statistic, p_value, effect_size
    """
    # Build group -> list of (pair_key, value)
    group_vals: dict[str, list[tuple[str | None, float]]] = {}

    for exp in experiments:
        g = str(exp.get(group_field, "") or "")
        if not g:
            continue

        v = _metric_value_at(exp, metric, value_mode)
        if v is None:
            continue

        pair_key = None
        if pairing_field:
            pair_key = str(exp.get(pairing_field, "") or "")

        group_vals.setdefault(g, []).append((pair_key, v))

    groups = sorted(group_vals.keys())
    rows = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ga, gb = groups[i], groups[j]
            a_list = group_vals[ga]
            b_list = group_vals[gb]

            # If pairing, intersect pair keys
            if pairing_field:
                a_map = {k: v for k, v in a_list if k is not None and k != ""}
                b_map = {k: v for k, v in b_list if k is not None and k != ""}
                common_keys = sorted(set(a_map.keys()) & set(b_map.keys()))

                a = np.array([a_map[k] for k in common_keys], dtype=float)
                b = np.array([b_map[k] for k in common_keys], dtype=float)

                n = len(common_keys)
                if n < 2:
                    continue

                diffs = a - b

                # paired tests
                t_stat, t_p = stats.ttest_rel(a, b, nan_policy="omit")
                try:
                    w_stat, w_p = stats.wilcoxon(diffs)
                except Exception:
                    w_stat, w_p = np.nan, np.nan

                # choose default test (Wilcoxon is robust)
                test_used = "wilcoxon_signed_rank"
                statistic = w_stat
                p_value = w_p

                effect = _cohens_dz_paired(diffs)

                rows.append(
                    {
                        "group_a": ga,
                        "group_b": gb,
                        "paired_on": pairing_field,
                        "n_pairs": n,
                        "value_mode": value_mode,
                        "test_used": test_used,
                        "statistic": statistic,
                        "p_value": p_value,
                        "effect_size": effect,
                        "alt_test_ttest_rel_p": t_p,
                    }
                )

            else:
                a = np.array([v for _, v in a_list], dtype=float)
                b = np.array([v for _, v in b_list], dtype=float)

                if len(a) < 2 or len(b) < 2:
                    continue

                # unpaired tests
                t_stat, t_p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                try:
                    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    u_stat, u_p = np.nan, np.nan

                test_used = "mann_whitney_u"
                statistic = u_stat
                p_value = u_p

                effect = _cohens_d_unpaired(a, b)

                rows.append(
                    {
                        "group_a": ga,
                        "group_b": gb,
                        "paired_on": None,
                        "n_a": int(len(a)),
                        "n_b": int(len(b)),
                        "value_mode": value_mode,
                        "test_used": test_used,
                        "statistic": statistic,
                        "p_value": p_value,
                        "effect_size": effect,
                        "alt_test_ttest_ind_p": t_p,
                    }
                )

    return pd.DataFrame(rows)
