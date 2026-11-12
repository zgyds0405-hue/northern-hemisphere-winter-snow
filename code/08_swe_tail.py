"""Quantify leverage from high SWE cells using fixed whole-record exclusions."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main():
    ds = xr.load_dataset(OUT / "processed/djf_winter_fields.nc", engine="h5netcdf")
    masks = np.load(OUT / "region_masks/fixed_masks.npz")
    swe = ds.snow_water_equivalent.values
    scf = ds.snow_cover_fraction.values
    area = ds.cell_area.values
    years = ds.winter_year.values
    time = (years - years.mean()) / 10
    x = sm.add_constant(time)
    peak = np.max(np.where(np.isfinite(swe), swe, -np.inf), axis=0)
    rows, pairs = [], []
    for region in masks.files:
        fixed = masks[region]
        for cut in [np.inf, 5000., 2000., 1000.]:
            retain = fixed & (peak <= cut)
            w = area[retain]
            f, s = scf[:, retain], swe[:, retain]
            amounts = {"scr_pct": 100 * f @ w / w.sum(), "swe_mm": s @ w / w.sum(),
                       "storage_a_km3": s @ w / 1e12, "storage_b_km3": (f*s) @ w / 1e12}
            wfull = area[fixed]
            a_full = swe[:, fixed] @ wfull / 1e12
            for name, y in amounts.items():
                fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True},
                                      use_t=True)
                lo, hi = fit.conf_int()[1]
                rows.append(dict(region=region, cut_djf_mm=cut, metric=name, n=len(years),
                                 n_cells=retain.sum(), excluded_cells=(fixed & ~retain).sum(),
                                 area_km2=w.sum()/1e6, mean=y.mean(), slope=fit.params[1],
                                 ci_low=lo, ci_high=hi, p_hac=fit.pvalues[1],
                                 share_a_mean_excluded=1-amounts["storage_a_km3"].mean()/a_full.mean()
                                 if a_full.mean() else np.nan))
            for name in ("swe_mm", "storage_b_km3"):
                fz = stats.zscore(amounts["scr_pct"], ddof=1)
                sz = stats.zscore(amounts[name], ddof=1)
                fit = sm.OLS(fz-sz, x).fit(cov_type="HAC",
                                           cov_kwds={"maxlags": 3, "use_correction": True}, use_t=True)
                lo, hi = fit.conf_int()[1]
                pairs.append(dict(region=region, cut_djf_mm=cut, metric=name, n=len(years),
                                  r_raw=stats.pearsonr(fz, sz).statistic,
                                  r_detrended=stats.pearsonr(sm.OLS(fz,x).fit().resid, sm.OLS(sz,x).fit().resid).statistic,
                                  standardized_slope_difference=fit.params[1],
                                  ci_low=lo, ci_high=hi, p_hac=fit.pvalues[1]))
    pd.DataFrame(rows).to_csv(OUT / "csv/swe_tail_sensitivity.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "csv/swe_tail_decoupling.csv", index=False)


if __name__ == "__main__":
    main()
