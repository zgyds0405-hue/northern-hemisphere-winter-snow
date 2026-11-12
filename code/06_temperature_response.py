"""Detrended temperature associations, slope heterogeneity and winter-scale lags."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from spatial_stats import (
    BLOCKS, CSV, GROUPS, HAC, acf1, block_indices, boot_interval, decadal_time,
    fdr, paired_corr_boot, residualize, write_table,
)

METRICS = {"scf": "percentage points per deg C", "swe": "mm per deg C"}


def response_model(y, temp, years):
    design = np.column_stack([np.ones(len(y)), temp, decadal_time(years)])
    return sm.OLS(y, design).fit()


def hac(model):
    return model.get_robustcov_results(cov_type="HAC", maxlags=HAC, use_correction=True, use_t=True)


def fit_response(series):
    fits, anomalies, lags, sensitivity = [], [], [], []
    for (family, group), subset in series.groupby(["group_type", "group"], sort=False):
        subset = subset.sort_values("winter_year")
        years = subset.winter_year.to_numpy()
        temp = subset.t2m.to_numpy()
        assert np.all(np.diff(years) == 1)
        for metric, unit in METRICS.items():
            y = subset[metric].to_numpy()
            assert np.isfinite(y).all() and np.isfinite(temp).all()
            y_d, temp_d = residualize(years, np.column_stack([y, temp]))[0].T
            raw = stats.pearsonr(y, temp)
            partial_r = stats.pearsonr(y_d, temp_d).statistic
            partial_t = partial_r * np.sqrt((len(y) - 3) / (1 - partial_r ** 2))
            partial_ci = np.nanquantile(paired_corr_boot(y_d, temp_d, 5), [.025, .975])
            ols = response_model(y, temp, years)
            model = hac(ols)
            estimate = model.params[1]
            influence = ols.get_influence().cooks_distance[0]
            loo = []
            base = {"group_type": family, "group": group, "metric": metric}
            for i, year in enumerate(years):
                keep = np.arange(len(y)) != i
                dropped = hac(response_model(y[keep], temp[keep], years[keep]))
                loo.append(dropped.params[1])
                sensitivity.append({**base, "omitted_winter": year, "slope": dropped.params[1],
                                    "ci_low": dropped.conf_int()[1, 0], "ci_high": dropped.conf_int()[1, 1],
                                    "p_hac": dropped.pvalues[1], "slope_change": dropped.params[1] - estimate,
                                    "full_model_cooks_distance": influence[i]})
            most_change = np.argmax(np.abs(np.array(loo) - estimate))
            fits.append({**base, "unit": unit, "n": len(y), "raw_r": raw.statistic,
                         "p_raw_iid": raw.pvalue, "detrended_r": partial_r,
                         "partial_r_year": partial_r, "p_partial_iid": 2 * stats.t.sf(abs(partial_t), len(y) - 3),
                         "partial_ci_low_block5": partial_ci[0], "partial_ci_high_block5": partial_ci[1],
                         "slope": estimate, "ci_low": model.conf_int()[1, 0], "ci_high": model.conf_int()[1, 1],
                         "p_hac": model.pvalues[1], "p_ols": ols.pvalues[1], "r2_full": model.rsquared,
                         "r2_detrended": partial_r ** 2, "residual_acf1": acf1(model.resid),
                         "durbin_watson": durbin_watson(model.resid),
                         "p_ljungbox_3": acorr_ljungbox(model.resid, lags=[3], return_df=True).lb_pvalue.iloc[0],
                         "p_breusch_pagan": het_breuschpagan(ols.resid, ols.model.exog)[1],
                         "max_cooks_distance": influence.max(), "max_cooks_year": years[np.argmax(influence)],
                         "loo_low": min(loo), "loo_high": max(loo), "loo_largest_change_year": years[most_change],
                         "loo_max_abs_change": abs(loo[most_change] - estimate)})
            for i, year in enumerate(years):
                anomalies.append({**base, "winter_year": year, "temp_anom": temp_d[i], "snow_anom": y_d[i]})
            for lag in range(-2, 3):
                overlap = (years - lag >= years[0]) & (years - lag <= years[-1])
                y_idx = np.flatnonzero(overlap)
                t_idx = y_idx - lag
                a, b = y_d[y_idx], temp_d[t_idx]
                r = stats.pearsonr(a, b)
                # k=0 uses the full year-adjusted model for exact slope inference.
                # Nonzero lag tests the overlap of the full-period detrended series.
                reg = model if lag == 0 else hac(sm.OLS(a, sm.add_constant(b)).fit())
                ci = np.nanquantile(paired_corr_boot(a, b, 5), [.025, .975])
                lags.append({**base, "lag": lag, "n": len(a), "r": r.statistic,
                             "ci_low_block5": ci[0], "ci_high_block5": ci[1], "p_hac": reg.pvalues[1],
                             "p_pearson_iid": r.pvalue, "snow_first_year": years[y_idx[0]],
                             "snow_last_year": years[y_idx[-1]], "temp_first_year": years[t_idx[0]],
                             "temp_last_year": years[t_idx[-1]]})
    model = fdr(pd.DataFrame(fits), p="p_hac", family=["group_type", "metric"], column="q_hac")
    lags = fdr(pd.DataFrame(lags), p="p_hac", family=["group_type", "group", "metric"], column="q_hac_lags5")
    fdr(lags, p="p_hac", column="q_hac_all_lags")
    return model, pd.DataFrame(anomalies), lags, pd.DataFrame(sensitivity)


def slope_draws(y, x, idx):
    y = y[idx]
    x = x[idx]
    y = y - y.mean(axis=1, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)
    return np.sum(y * x, axis=1) / np.sum(x ** 2, axis=1)


def interactions(anomalies, fits):
    direct, omnibus = [], []
    for family, groups in GROUPS.items():
        if len(groups) < 2:
            continue
        for metric in METRICS:
            data = anomalies[(anomalies.group_type == family) & (anomalies.metric == metric)]
            y = data.pivot(index="winter_year", columns="group", values="snow_anom")[groups].to_numpy()
            x = data.pivot(index="winter_year", columns="group", values="temp_anom")[groups].to_numpy()
            estimates = fits[(fits.group_type == family) & (fits.metric == metric)].set_index("group").loc[groups, "slope"].to_numpy()
            np.testing.assert_allclose(estimates, np.sum(x * y, axis=0) / np.sum(x ** 2, axis=0))
            for length in BLOCKS:
                idx = block_indices(len(y), length)
                reps = slope_draws(y, x, idx)
                for i, j in combinations(range(len(groups)), 2):
                    delta = estimates[i] - estimates[j]
                    direct.append({"group_type": family, "metric": metric, "left": groups[i], "right": groups[j],
                                   "difference": delta, "block": length, "n": len(y),
                                   **boot_interval(reps[:, i] - reps[:, j], delta)})
                # Joint test of all interaction terms against the first group's slope.
                centered = reps[:, 1:] - reps[:, :1]
                theta = estimates[1:] - estimates[0]
                cov_inv = np.linalg.pinv(np.cov(centered, rowvar=False))
                wald = theta @ cov_inv @ theta
                null = centered - theta
                null_wald = np.einsum("ij,jk,ik->i", null, cov_inv, null)
                p = (1 + np.sum(null_wald >= wald)) / (1 + len(reps))
                omnibus.append({"group_type": family, "metric": metric, "block": length, "n": len(y),
                                "df": len(groups) - 1, "wald": wald, "p_boot": p})
    d = fdr(pd.DataFrame(direct), p="p_boot", family=["group_type", "metric", "block"], column="q_boot")
    o = fdr(pd.DataFrame(omnibus), p="p_boot", family=["block"], column="q_boot")
    return d, o


def main():
    series = pd.read_csv(CSV / "regional_latitude_metrics.csv")
    fits, anomalies, lags, loo = fit_response(series)
    direct, omnibus = interactions(anomalies, fits)
    for table, stem in ((fits, "temperature_response"), (anomalies, "temperature_anomalies"),
                        (lags, "temperature_lags"), (loo, "temperature_leave_one_out"),
                        (direct, "temperature_interaction_contrasts"), (omnibus, "temperature_interaction_omnibus")):
        write_table(table, stem, "climate")
    print(fits[["group", "metric", "slope", "ci_low", "ci_high", "partial_r_year",
                "p_hac", "q_hac", "loo_low", "loo_high"]].to_string(index=False))
    print(omnibus[omnibus.block == 5].to_string(index=False))
    print(lags[(lags.lag != 0) & (lags.q_hac_all_lags <= .05)].to_string(index=False))


if __name__ == "__main__":
    main()
