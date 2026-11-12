"""Sensitivity checks using only the report's annual source-data tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "csv"


def weather_checks():
    weather = pd.read_csv(CSV / "regional_latitude_metrics.csv")
    rows = []
    for (group_type, group), frame in weather.groupby(["group_type", "group"], sort=False):
        frame = frame.sort_values("winter_year")
        year = frame.winter_year.to_numpy()
        step = np.where(year == 2004, 60 / 91, year > 2004)
        nuisance = sm.add_constant(np.column_stack(((year - year.mean()) / 10, step)))
        model = np.column_stack((nuisance, frame.t2m))
        temp_anom = sm.OLS(frame.t2m, nuisance).fit().resid
        snow_anom = {key: sm.OLS(frame[key], nuisance).fit().resid for key in ("scf", "swe")}
        co_variation = stats.pearsonr(snow_anom["scf"], snow_anom["swe"]).statistic
        for key in snow_anom:
            fit = sm.OLS(frame[key].to_numpy(), model).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True}, use_t=True
            )
            lo, hi = fit.conf_int()[-1]
            rows.append({
                "group_type": group_type, "group": group, "metric": key, "n": len(year),
                "unit": "pp/degC" if key == "scf" else "mm/degC",
                "slope": fit.params[-1], "ci_low": lo, "ci_high": hi, "p_hac": fit.pvalues[-1],
                "partial_r_year_step": stats.pearsonr(snow_anom[key], temp_anom).statistic,
                "scf_swe_partial_r_year_step": co_variation,
            })
    table = pd.DataFrame(rows)
    table["q_hac"] = table.groupby(["group_type", "metric"]).p_hac.transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    table.to_csv(HERE / "temperature_assimilation_checks.csv", index=False)
    return table


def calendar_hac3(year, values):
    """Omit the transition winter without converting the gap to one lag."""
    year_scale = (year - year.mean()) / 10
    design = sm.add_constant(np.column_stack((year_scale, year > 2004)))
    keep = (year != 2004) & np.isfinite(values)
    x = design[keep]
    fit = sm.OLS(values[keep], x).fit()
    score = np.zeros_like(design)
    score[keep] = x * fit.resid[:, None]
    meat = score.T @ score
    for lag in range(1, 4):
        term = score[lag:].T @ score[:-lag]
        meat += (1 - lag / 4) * (term + term.T)
    bread = np.linalg.inv(x.T @ x)
    n = keep.sum()
    df = n - x.shape[1]
    covariance = bread @ meat @ bread * n / df
    se = np.sqrt(covariance[1, 1])
    slope = fit.params[1]
    lo, hi = slope + stats.t.ppf(.975, df) * se * np.array([-1, 1])
    return {"n": n, "slope_deg_per_decade": slope, "ci_low": lo, "ci_high": hi,
            "p_hac": 2 * stats.t.sf(abs(slope / se), df)}


def snowline_checks():
    boundary = pd.read_csv(CSV / "snowline_annual.csv").query("rule == 'main'")
    rows = []
    for (threshold, region, support), frame in boundary.groupby(["threshold", "region", "support"]):
        frame = frame.sort_values("winter_year")
        rows.append({"threshold": threshold, "region": region, "support": support,
                     **calendar_hac3(frame.winter_year.to_numpy(), frame.median_degN.to_numpy())})
    table = pd.DataFrame(rows)
    table["q_hac"] = table.groupby("support").p_hac.transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    table.to_csv(HERE / "snowline_assimilation_checks.csv", index=False)
    return table


if __name__ == "__main__":
    print(weather_checks().query("group == 'NH'").to_string(index=False))
    print(snowline_checks().query("region == 'NH' and threshold == .50").to_string(index=False))
