"""Aggregate fixed-domain DJF metrics and calculate temporal inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CSV = OUT / "csv"
AGENT = OUT / "agent_trend"
PROCESSED = OUT / "processed"
CONFIG = json.loads((OUT / "analysis_config.yaml").read_text(encoding="utf-8"))
VARIABLES = CONFIG["variables"]

UNITS = {
    "sca_km2": "km2", "scr_pct": "percentage points", "scf_fraction": "fraction",
    "swe_mm": "mm", "storage_a_km3": "km3", "storage_b_km3": "km3", "t2m_c": "degC",
    "equivalent_a_mm": "mm", "equivalent_b_mm": "mm",
}


def save_table(table, name):
    for directory in (CSV, AGENT):
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / name, index=False)


def seasonal_metrics(data, spatial_mask, coverage):
    """Area weights are provided in m². A/B storage uses like-for-like coverage."""
    area = data.cell_area.values
    scf, swe, t2m = (data[name].values[:, spatial_mask] for name in VARIABLES.values())
    weights = area[spatial_mask]
    joint = np.isfinite(scf) & np.isfinite(swe)
    reference_area = weights.sum()

    def sum_valid(value, valid):
        numerator = np.where(valid, value, 0) @ weights
        return np.where(valid.any(axis=1), numerator, np.nan)

    def weighted_mean(value):
        finite = np.isfinite(value)
        return sum_valid(value, finite) / np.where(finite.any(axis=1), finite @ weights, np.nan)

    available_scf = np.isfinite(scf)
    sca_m2 = sum_valid(scf, available_scf)
    joint_sca_m2 = sum_valid(scf, joint)
    a_mm_m2 = sum_valid(swe, joint)
    b_mm_m2 = sum_valid(scf * swe, joint)
    metrics = pd.DataFrame({
        "winter_year": data.winter_year.values,
        "coverage": coverage,
        "mask_n": spatial_mask.sum(),
        "mask_area_km2": reference_area / 1e6,
        "scf_valid_n": available_scf.sum(axis=1),
        "swe_valid_n": np.isfinite(swe).sum(axis=1),
        "t2m_valid_n": np.isfinite(t2m).sum(axis=1),
        "scf_swe_valid_n": joint.sum(axis=1),
        "scf_valid_area_km2": available_scf @ weights / 1e6,
        "swe_valid_area_km2": np.isfinite(swe) @ weights / 1e6,
        "t2m_valid_area_km2": np.isfinite(t2m) @ weights / 1e6,
        "scf_swe_valid_area_km2": joint @ weights / 1e6,
        "sca_km2": sca_m2 / 1e6,
        "scr_pct": weighted_mean(scf) * 100,
        "scf_fraction": weighted_mean(scf),
        "swe_mm": weighted_mean(swe),
        "storage_a_km3": a_mm_m2 / 1e12,
        "storage_b_km3": b_mm_m2 / 1e12,
        "t2m_c": weighted_mean(t2m),
        "equivalent_a_mm": np.divide(a_mm_m2, joint_sca_m2, out=np.full_like(a_mm_m2, np.nan), where=joint_sca_m2 > 0),
        "equivalent_b_mm": np.divide(b_mm_m2, joint_sca_m2, out=np.full_like(b_mm_m2, np.nan), where=joint_sca_m2 > 0),
    })
    return metrics


def trend(year, value):
    """OLS, Newey–West HAC, tied MK and Sen; slopes are expressed per decade."""
    valid = np.isfinite(year) & np.isfinite(value)
    year, value = np.asarray(year)[valid], np.asarray(value)[valid]
    n = year.size
    if n < 5:
        raise ValueError("A winter series requires at least five valid winter observations.")
    x = (year - year.mean()) / 10
    if np.ptp(value) == 0:
        return dict(n=n, n_eff=n, mean=value[0], slope=0, ci_low=0, ci_high=0,
                    p_hac=1, ols_ci_low=0, ols_ci_high=0, p_ols=1, r2=np.nan, rho1=np.nan,
                    mk_s=0, mk_z=0, mk_p=1, sen_slope=0, sen_low=0, sen_high=0)
    ols = sm.OLS(value, sm.add_constant(x)).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=CONFIG["hac_maxlags"],
                                   use_correction=True, use_t=True)
    ci, ci_ols = hac.conf_int()[1], ols.conf_int()[1]
    a, b = np.triu_indices(n, 1)
    mk_s = np.sign(value[b] - value[a]).sum()
    _, ties = np.unique(value, return_counts=True)
    variance = (n * (n - 1) * (2 * n + 5) - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18
    mk_z = (mk_s - np.sign(mk_s)) / np.sqrt(variance) if variance else 0
    sen = stats.theilslopes(value, x)
    pair = np.diff(year) == 1
    rho1 = np.corrcoef(ols.resid[:-1][pair], ols.resid[1:][pair])[0, 1] if pair.sum() > 2 else np.nan
    n_eff = n * (1 - max(rho1, 0)) / (1 + max(rho1, 0)) if np.isfinite(rho1) else np.nan
    return dict(n=n, n_eff=n_eff, mean=value.mean(), slope=hac.params[1],
                ci_low=ci[0], ci_high=ci[1], p_hac=hac.pvalues[1], ols_ci_low=ci_ols[0],
                ols_ci_high=ci_ols[1], p_ols=ols.pvalues[1], r2=ols.rsquared, rho1=rho1,
                mk_s=int(mk_s), mk_z=mk_z, mk_p=2 * stats.norm.sf(abs(mk_z)),
                sen_slope=sen.slope, sen_low=sen.low_slope, sen_high=sen.high_slope)


def circular_indices(rng, n, block, draws):
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block)), 1))
    return ((starts + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def residual_bootstrap(x, values, block, draws, seed):
    """Jointly resample circular blocks of residual vectors around fitted trends."""
    slope_weights = x / (x @ x)
    beta = slope_weights @ values
    residuals = values - values.mean(axis=0) - x[:, None] * beta
    ids = circular_indices(np.random.default_rng(seed), len(x), block, draws)
    resampled = np.einsum("n,bnk->bk", slope_weights, residuals[ids])
    return beta + resampled, residuals


def high_swe_sensitivity(data, mask):
    """Remove a fixed, nested set of high-SWE grid cells for a stress test."""
    climatology = data[VARIABLES["swe"]].values[:, mask].mean(axis=0)
    rows = []
    for cutoff in (1000, 2000, 5000):
        keep = mask.copy()
        keep[mask] = climatology < cutoff
        frame = seasonal_metrics(data, keep, f"climatology_lt_{cutoff}_mm")
        for metric in ("scr_pct", "swe_mm", "storage_a_km3", "storage_b_km3", "t2m_c"):
            rows.append({"climatology_cutoff_mm": cutoff, "excluded_grid_n": (mask & ~keep).sum(),
                         "excluded_area_km2": data.cell_area.values[mask & ~keep].sum() / 1e6,
                         "metric": metric, "units_per_decade": UNITS[metric],
                         **trend(frame.winter_year.to_numpy(), frame[metric].to_numpy())})
    save_table(pd.DataFrame(rows), "global_high_swe_sensitivity.csv")


def assimilation_step_sensitivity(frame):
    """Treat the reported 2004 ERA5 discontinuity as a level-shift sensitivity."""
    year = frame.winter_year.to_numpy()
    x = (year - year.mean()) / 10
    # For an onset in Jan 2004, only the last 60 of DJF 2004's 91 days are exposed.
    January_2004 = np.where(year == 2004, 60 / 91, (year > 2004).astype(float))
    breaks = {"jan_2004_weighted": January_2004, **{f"djf_{start}": (year >= start).astype(float)
              for start in (2004, 2005, 2006)}}
    rows = []
    for metric in ("scr_pct", "swe_mm", "storage_a_km3", "storage_b_km3", "t2m_c"):
        value = frame[metric].to_numpy()
        for name, step in breaks.items():
            design = sm.add_constant(np.column_stack([x, step]))
            fit = sm.OLS(value, design).fit()
            hac = fit.get_robustcov_results(cov_type="HAC", maxlags=CONFIG["hac_maxlags"],
                                           use_correction=True, use_t=True)
            ci = hac.conf_int()
            rows.append({"metric": metric, "model": name, "units_per_decade": UNITS[metric],
                         "n": len(year), "first_year": year.min(), "last_year": year.max(),
                         "slope": hac.params[1], "ci_low": ci[1, 0], "ci_high": ci[1, 1],
                         "p_hac": hac.pvalues[1], "step": hac.params[2],
                         "step_ci_low": ci[2, 0], "step_ci_high": ci[2, 1],
                         "step_p_hac": hac.pvalues[2], "r2": fit.rsquared})
        for name, keep in (("pre_1980_2003", year < 2004), ("post_2005_2022", year > 2004)):
            rows.append({"metric": metric, "model": name, "units_per_decade": UNITS[metric],
                         "first_year": year[keep].min(), "last_year": year[keep].max(),
                         **trend(year[keep], value[keep])})
    save_table(pd.DataFrame(rows), "global_assimilation_step_sensitivity.csv")


def global_analysis(data):
    mask = data.analysis_mask.values.astype(bool)
    complete = mask.copy()
    for variable in VARIABLES:
        complete &= data[f"complete_{variable}"].values.astype(bool)
    main = seasonal_metrics(data, mask, "fixed_mask")
    strict = seasonal_metrics(data, complete, "joint_complete")
    all_metrics = pd.concat([main, strict], ignore_index=True)
    for metric in ("swe_mm", "storage_a_km3", "storage_b_km3"):
        all_metrics[f"d_{metric}"] = all_metrics.groupby("coverage", sort=False)["sca_km2"].transform(
            lambda y: (y - y.mean()) / y.std(ddof=1)
        ) - all_metrics.groupby("coverage", sort=False)[metric].transform(lambda y: (y - y.mean()) / y.std(ddof=1))
    save_table(main.merge(all_metrics[all_metrics.coverage == "fixed_mask"][["winter_year", "d_swe_mm", "d_storage_a_km3", "d_storage_b_km3"]], on="winter_year"), "global_metrics.csv")
    save_table(all_metrics, "global_metrics_coverage_sensitivity.csv")
    trends, bootstrap, decoupling = [], [], []
    for label, frame in all_metrics.groupby("coverage", sort=False):
        year = frame.winter_year.to_numpy()
        for metric, units in UNITS.items():
            record = trend(year, frame[metric].to_numpy())
            trends.append({"coverage": label, "metric": metric, "units_per_decade": units, **record})

    values = main[list(UNITS)].to_numpy()
    x = (main.winter_year.to_numpy() - main.winter_year.mean()) / 10
    n = len(x)
    for block in CONFIG["bootstrap_blocks"]:
        simulated, _ = residual_bootstrap(x, values, block, CONFIG["bootstrap_draws"], CONFIG["bootstrap_seed"])
        for j, metric in enumerate(UNITS):
            low, high = np.quantile(simulated[:, j], [.025, .975])
            bootstrap.append({"metric": metric, "block_winters": block, "n": n, "draws": len(simulated),
                              "ci_low": low, "ci_high": high})
    for metric in ("swe_mm", "storage_a_km3", "storage_b_km3"):
        raw = main[["sca_km2", metric]].to_numpy()
        standard = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
        observed = (x @ standard[:, 0] - x @ standard[:, 1]) / (x @ x)
        for block in CONFIG["bootstrap_blocks"]:
            simulated, residuals = residual_bootstrap(x, standard, block, CONFIG["bootstrap_draws"],
                                                      CONFIG["bootstrap_seed"])
            differences = simulated[:, 0] - simulated[:, 1]
            low, high = np.quantile(differences, [.025, .975])
            p_boot = (1 + (np.abs(differences - observed) >= abs(observed)).sum()) / (len(differences) + 1)
            decoupling.append({"swe_metric": metric, "n": n, "block_winters": block,
                               "correlation_raw": np.corrcoef(raw.T)[0, 1],
                               "correlation_detrended": np.corrcoef(residuals.T)[0, 1],
                               "difference_sd_per_decade": observed, "ci_low": low, "ci_high": high,
                               "p_boot_centered": p_boot, "draws": len(differences)})
    save_table(pd.DataFrame(trends), "trend_global.csv")
    save_table(pd.DataFrame(bootstrap), "global_bootstrap.csv")
    save_table(pd.DataFrame(decoupling), "global_decoupling.csv")
    high_swe_sensitivity(data, mask)
    assimilation_step_sensitivity(main)
    print(pd.DataFrame(trends).iloc[:len(UNITS)].to_string(index=False))
    print(pd.DataFrame(decoupling).to_string(index=False))


if __name__ == "__main__":
    with xr.open_dataset(PROCESSED / "djf_winter_fields.nc", engine="h5netcdf") as djf:
        global_analysis(djf)
