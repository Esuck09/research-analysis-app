# analysis/bootstrap.py

from __future__ import annotations
import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of a 1D array.
    """
    rng = np.random.default_rng(random_state)
    n = len(values)

    if n < 2:
        return (np.nan, np.nan)

    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1.0 - alpha)

    return float(lower), float(upper)


def bootstrap_curve_ci(
    stacked: pd.DataFrame,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Compute bootstrap CI per epoch.

    stacked: DataFrame of shape (epochs × runs)
    """
    lowers = []
    uppers = []

    for _, row in stacked.iterrows():
        vals = row.dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(
            vals,
            n_boot=n_boot,
            ci=ci,
            random_state=random_state,
        )
        lowers.append(lo)
        uppers.append(hi)

    return pd.DataFrame(
        {
            "ci_lower": lowers,
            "ci_upper": uppers,
        }
    )
