"""Exploratory January 2004 sensitivity for the NH boundary and temperature associations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "output" / "csv"
AGENT = ROOT / "output" / "agent_trend"
CONFIG = json.loads((ROOT / "output" / "analysis_config.yaml").read_text(encoding="utf-8"))


def fit(y, x):
    return sm.OLS(y, x).fit().get_robustcov_results(
        cov_type="HAC", maxlags=CONFIG["hac_maxlags"], use_correction=True, use_t=True
    )


def coefficients(model, index):
    low, high = model.conf_int()[index]
    return dict(slope=model.params[index], ci_low=low, ci_high=high, p_hac=model.pvalues[index])


def main():
    snowline = pd.read_csv(CSV / "snowline_annual.csv")
    snowline = snowline[(snowline.region == "NH") & (snowline.support == "available") &
                        (snowline.rule == "main") & (snowline.threshold == .5)].sort_values("winter_year")
    global_data = pd.read_csv(CSV / "global_metrics.csv").sort_values("winter_year")
    year = global_data.winter_year.to_numpy()
    np.testing.assert_array_equal(year, snowline.winter_year.to_numpy())
    boundary = snowline.median_degN.to_numpy()
    assert np.isfinite(boundary).all()
    x = (year - year.mean()) / 10
    # Mortimer et al. (2020, Fig. 7) locate the ERA5 discontinuity in January 2004.
    # Jan and Feb 2004 contribute 60 of the 91 days in the DJF 2004 mean.
    step = np.where(year == 2004, 60 / 91, (year > 2004).astype(float))
    samples = len(year)
    temperature = global_data.t2m_c.to_numpy()
    results = []
    for label, design in [("year_only", sm.add_constant(x)),
                          ("year_and_jan_2004_step", sm.add_constant(np.column_stack([x, step])))]:
        fit_boundary = fit(boundary, design)
        entry = dict(metric="boundary_latitude", model=label, units="degN per decade", n=samples,
                     r2=fit_boundary.rsquared, **coefficients(fit_boundary, 1))
        if design.shape[1] == 3:
            entry.update({"step": fit_boundary.params[2], "step_p_hac": fit_boundary.pvalues[2]})
        results.append(entry)
        design_temp = np.column_stack([design, temperature])
        temp_residual = fit(temperature, design).resid
        for metric, units in [("scr_pct", "percentage points per degC"), ("swe_mm", "mm per degC")]:
            value = global_data[metric].to_numpy()
            model = fit(value, design_temp)
            partial_r = np.corrcoef(temp_residual, fit(value, design).resid)[0, 1]
            results.append(dict(metric=metric, model=label, units=units, n=samples, r2=model.rsquared,
                                partial_r=partial_r, **coefficients(model, -1)))
    table = pd.DataFrame(results)
    for dest in (CSV, AGENT):
        table.to_csv(dest / "snowline_temperature_assimilation_sensitivity.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
