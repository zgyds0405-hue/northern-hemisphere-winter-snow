"""Shared area-series and resampling utilities for RQ2 and RQ3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
CSV = OUTPUT / "csv"
CONFIG = json.loads((OUTPUT / "analysis_config.yaml").read_text(encoding="utf-8"))
REGIONS = list(CONFIG["windows"])
BANDS = list(CONFIG["latitude_bands"])
GROUPS = {"hemisphere": ["NH"], "window": REGIONS, "latitude": BANDS}
METRICS = {
    "scf": "percentage points", "swe": "mm", "t2m": "deg C",
    "sca": "km2", "volume_a": "km3", "volume_b": "km3",
}
SEED = CONFIG["bootstrap_seed"]
DRAWS = CONFIG["bootstrap_draws"]
BLOCKS = CONFIG["bootstrap_blocks"]
HAC = CONFIG["hac_maxlags"]


def write_table(table, stem, agent):
    target = f"{stem}.csv"
    table.to_csv(CSV / target, index=False, float_format="%.12g")
    table.to_csv(OUTPUT / f"agent_{agent}" / target, index=False, float_format="%.12g")


def fdr(table, p="p", family=None, column="q_bh"):
    table[column] = np.nan
    subgroups = [table] if family is None else (
        group for _, group in table.groupby(family, sort=False)
    )
    for group in subgroups:
        valid = group.index[np.isfinite(group[p])]
        if len(valid):
            table.loc[valid, column] = multipletests(table.loc[valid, p], method="fdr_bh")[1]
    return table


def decadal_time(years):
    years = np.asarray(years, dtype=float)
    return (years - years.mean()) / 10


def residualize(years, observations):
    x = sm.add_constant(decadal_time(years))
    values = np.asarray(observations)
    fit = x @ np.linalg.lstsq(x, values, rcond=None)[0]
    return values - fit, fit


def zscore(x):
    x = np.asarray(x)
    return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)


def acf1(values):
    return float(np.corrcoef(values[:-1], values[1:])[0, 1]) if np.std(values) else np.nan


def mk_test(y):
    delta = y[None, :] - y[:, None]
    score = np.sign(delta[np.triu_indices(y.size, k=1)]).sum()
    n = y.size
    _, ties = np.unique(y, return_counts=True)
    variance = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18
    z = (score - np.sign(score)) / np.sqrt(variance) if variance and score else 0
    return score, 2 * stats.norm.sf(abs(z))


def trend(years, y):
    good = np.isfinite(y)
    years, y = np.asarray(years)[good], np.asarray(y)[good]
    time = decadal_time(years)
    ols = sm.OLS(y, sm.add_constant(time)).fit()
    model = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC, use_correction=True, use_t=True)
    sen = stats.theilslopes(y, time)
    score, p_mk = mk_test(y)
    r1 = acf1(model.resid)
    n_eff = np.clip(y.size * (1 - r1) / (1 + r1), 3, y.size) if np.isfinite(r1) else y.size
    return {"slope": model.params[1], "ci_low": model.conf_int()[1, 0],
            "ci_high": model.conf_int()[1, 1], "p_hac": model.pvalues[1],
            "p_ols": ols.pvalues[1], "ols_ci_low": ols.conf_int()[1, 0],
            "ols_ci_high": ols.conf_int()[1, 1], "r2": model.rsquared,
            "n": y.size, "n_eff_ar1_approx": n_eff, "residual_acf1": r1,
            "mk_s": score, "p_mk": p_mk, "sen": sen.slope,
            "sen_ci_low": sen.low_slope, "sen_ci_high": sen.high_slope}


def block_indices(n, length):
    rng = np.random.default_rng(SEED + 1000 * n + length)
    starts = rng.integers(n, size=(DRAWS, int(np.ceil(n / length))))
    return ((starts[..., None] + np.arange(length)) % n).reshape(DRAWS, -1)[:, :n]


def paired_slope_boot(years, values, length):
    """Circular moving blocks of synchronous OLS residuals, fit trend held in place."""
    residual, fit = residualize(years, values)
    x = decadal_time(years)
    idx = block_indices(x.size, length)
    return np.einsum("t,btk->bk", x, fit[None, :, :] + residual[idx]) / (x @ x)


def paired_corr_boot(left, right, length):
    idx = block_indices(len(left), length)
    a, b = left[idx], right[idx]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    divisor = np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))
    return np.divide(np.sum(a * b, axis=1), divisor, out=np.full(a.shape[0], np.nan), where=divisor > 0)


def boot_interval(draws, estimate):
    ci = np.nanquantile(draws, [.025, .975])
    centered = draws - estimate
    p = (1 + np.sum(np.abs(centered[np.isfinite(centered)]) >= abs(estimate))) / (1 + np.isfinite(centered).sum())
    return {"ci_low": ci[0], "ci_high": ci[1], "p_boot": p}
