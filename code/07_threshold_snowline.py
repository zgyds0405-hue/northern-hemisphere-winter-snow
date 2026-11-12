"""Extract valid adjacent-land SCF threshold crossings and quantify shifts."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, sparse, stats
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CSV = OUT / "csv"
AGENT = OUT / "agent_snowline"
SCF = "snow_cover_fraction"
RULES = {"main": (9, 2, False), "unfiltered": (1, 1, False),
         "strict": (25, 3, False), "interior": (9, 2, True)}
BOOTSTRAP_LONGITUDE_DEGREES = (5, 10, 20)


def periodic_component_sizes(covered):
    """Sizes of 4-neighbour covered components with the longitude seam joined."""
    labels, count = ndimage.label(covered)
    left, right = labels[:, 0], labels[:, -1]
    joins = (left > 0) & (right > 0)
    graph = sparse.coo_matrix(
        (np.ones(joins.sum()), (left[joins], right[joins])),
        shape=(count + 1, count + 1),
    ).tocsr()
    _, merged = connected_components(graph, directed=False)
    component = merged[labels]
    sizes = np.bincount(component.ravel())
    return np.where(covered, sizes[component], 0)


def crossing_latitudes(field, lat, theta, sizes, min_cells, north_run, interior, lo, hi):
    """Southernmost retained crossing per longitude and count of all candidates."""
    south, north = field[:-1], field[1:]
    candidate = ((south < theta) & (north >= theta)
                 & (sizes[1:] >= min_cells)
                 & (lat[:-1, None] >= lo) & (lat[1:, None] <= hi))
    if interior:
        # A conservative coastal/missing-cell check: both adjacent quadrilaterals
        # must be fully supported, with periodic east/west neighbours.
        finite_pair = np.isfinite(south) & np.isfinite(north)
        candidate &= np.roll(finite_pair, 1, axis=1) & np.roll(finite_pair, -1, axis=1)
    # The above-threshold run is entirely on valid cells within the window.
    for step in range(1, north_run):
        check = np.zeros_like(candidate)
        check[:-step] = ((field[1 + step:] >= theta)
                         & (lat[1 + step:, None] <= hi))
        candidate &= check
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.divide(theta - south, north - south, out=np.zeros_like(south),
                             where=candidate)
    all_cross = np.where(candidate, lat[:-1, None] + fraction * np.diff(lat)[:, None], np.inf)
    equatorward = all_cross.min(axis=0)
    return np.where(np.isfinite(equatorward), equatorward, np.nan), candidate.sum(axis=0)


def summarize(x):
    vals = x[np.isfinite(x)]
    if not vals.size:
        return {"n_longitudes": 0, "q25_degN": np.nan, "median_degN": np.nan, "q75_degN": np.nan}
    q25, med, q75 = np.quantile(vals, [.25, .5, .75])
    return {"n_longitudes": vals.size, "q25_degN": q25, "median_degN": med, "q75_degN": q75}


def hac_trend(years, values, maxlags):
    """Fit OLS with calendar-lag HAC; preserve actual years in any gaps."""
    valid = np.isfinite(values)
    result = {"n_winters": int(valid.sum())}
    if valid.sum() < 8:
        return result, np.full((len(years), 3), np.nan)
    design = sm.add_constant((years - years.mean()) / 10.)
    x = design[valid]
    fit = sm.OLS(values[valid], x).fit()
    yhat = design @ fit.params
    # Zero scores for missing winters allow exact year lags in the HAC sum.
    score = np.zeros_like(design)
    score[valid] = x * fit.resid[:, None]
    meat = score.T @ score
    for k in range(1, maxlags + 1):
        cross = score[k:].T @ score[:-k]
        meat += (1 - k / (maxlags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    n = int(valid.sum())
    cov = (bread @ meat @ bread) * n / (n - 2)
    se = np.sqrt(max(cov[1, 1], 0.))
    critical = stats.t.ppf(.975, n - 2)
    slope = fit.params[1]
    lo, hi = slope + critical * se * np.array([-1., 1.])
    p = float(2 * stats.t.sf(abs(slope) / se, n - 2)) if se else (1. if slope == 0 else 0.)
    ci_half = critical * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, cov, design), 0.))
    result.update(slope_deg_per_decade=slope, hac_ci95_low=lo, hac_ci95_high=hi, hac_p=p,
                  r_squared=fit.rsquared, sen_deg_per_decade=10 * stats.theilslopes(values[valid], years[valid]).slope)
    return result, np.c_[yhat, yhat - ci_half, yhat + ci_half]


def paired_longitude_interval(shift, draws, length, seed, circular):
    """Percentile interval for median paired shifts, preserving longitude blocks."""
    width = len(shift)
    if np.isfinite(shift).sum() < length:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(width / length))
    valid_starts = width if circular else width - length + 1
    start = rng.integers(0, valid_starts, size=(draws, n_blocks))
    index = ((start[:, :, None] + np.arange(length)) % width).reshape(draws, -1)[:, :width]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replicates = np.nanmedian(shift[index], axis=1)
    replicates = replicates[np.isfinite(replicates)]
    if not replicates.size:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(replicates, [.025, .975])
    return lo, hi, len(replicates)


def main():
    config = json.loads((OUT / "analysis_config.yaml").read_text(encoding="utf-8"))
    with xr.open_dataset(OUT / "processed/djf_winter_fields.nc", engine="h5netcdf") as data:
        years, lat, lon = data.winter_year.values, data.latitude.values, data.longitude.values
        field = data[SCF].values
        mask = data.analysis_mask.values.astype(bool)
        assert np.isfinite(field[:, mask]).all() and np.isnan(field[:, ~mask]).all()
        assert np.all(np.diff(lat) == 1) and np.all(np.diff(lon) == 1)
    periods = {label: field[(years >= config[label][0]) & (years <= config[label][1])].mean(axis=0)
               for label in ("early", "late")}
    AGENT.mkdir(exist_ok=True)
    np.savez_compressed(AGENT / "snowline_period_fields.npz", latitude=lat, longitude=lon,
                        **periods, difference=periods["late"] - periods["early"])
    windows = {"NH": [0, 90, -180, 180], **config["windows"]}
    slots = {name: (lo, hi, (lon >= xlo) & (lon < xhi))
             for name, (lo, hi, xlo, xhi) in windows.items()}
    annual_rows, period_rows, trend_rows, shift_rows, lon_rows, prediction_rows = [], [], [], [], [], []
    all_fields = dict(zip(years.tolist(), field))
    all_fields.update(periods)
    for theta in config["thresholds"]:
        sizes_by_slot = {label: periodic_component_sizes(image >= theta) for label, image in all_fields.items()}
        for rule, (min_cells, north_run, interior) in RULES.items():
            for region, (lo, hi, columns) in slots.items():
                best, counts = {}, {}
                for label, image in all_fields.items():
                    lats, nums = crossing_latitudes(image, lat, theta, sizes_by_slot[label],
                                                   min_cells, north_run, interior, lo, hi)
                    best[label], counts[label] = lats[columns], nums[columns]
                metadata = {"threshold": theta, "rule": rule, "region": region}
                annual = np.stack([best[y] for y in years])
                complete = np.isfinite(annual).all(axis=0)
                if rule == "main":
                    for slot in all_fields:
                        lon_rows.extend({**metadata, "slot": slot, "longitude": x,
                                         "southernmost_latitude": l, "candidate_count": n}
                                        for x, l, n in zip(lon[columns], best[slot], counts[slot]))
                for support, chosen in [("available", np.ones(annual.shape[1], bool)),
                                         ("complete_43_winters", complete)]:
                    meds = []
                    for y in years:
                        row = summarize(best[y][chosen])
                        meds.append(row["median_degN"])
                        annual_rows.append({**metadata, "support": support, "winter_year": y,
                                            "n_candidates": counts[y][chosen].sum(), **row})
                    fit, predicted = hac_trend(years, np.array(meds), config["hac_maxlags"])
                    trend_rows.append({**metadata, "support": support, **fit,
                                       "n_longitudes_min": min(np.isfinite(annual[:, chosen]).sum(axis=1)),
                                       "n_longitudes_max": max(np.isfinite(annual[:, chosen]).sum(axis=1))})
                    for y, (mean, low, high) in zip(years, predicted):
                        prediction_rows.append({**metadata, "support": support, "winter_year": y,
                                                "fitted_degN": mean, "fitted_ci95_low": low, "fitted_ci95_high": high})
                paired = np.isfinite(best["early"]) & np.isfinite(best["late"])
                for slot in periods:
                    for support, chosen in [("available", np.ones(annual.shape[1], bool)), ("paired", paired)]:
                        period_rows.append({**metadata, "period": slot, "support": support,
                                            "n_candidates": counts[slot][chosen].sum(),
                                            **summarize(best[slot][chosen])})
                delta = best["late"] - best["early"]
                paired_summary = summarize(delta)
                diff_medians = np.median(best["late"][paired]) - np.median(best["early"][paired]) if paired.any() else np.nan
                for block in BOOTSTRAP_LONGITUDE_DEGREES:
                    a, b, n_boot = paired_longitude_interval(delta, config["bootstrap_draws"], block,
                                                            config["bootstrap_seed"], region == "NH")
                    shift_rows.append({**metadata, "longitude_block_deg": block,
                                       "paired_n_longitudes": paired_summary["n_longitudes"],
                                       "paired_median_shift_deg": paired_summary["median_degN"],
                                       "paired_shift_q25_deg": paired_summary["q25_degN"],
                                       "paired_shift_q75_deg": paired_summary["q75_degN"],
                                       "difference_paired_medians_deg": diff_medians,
                                       "spatial_bootstrap_ci95_low": a, "spatial_bootstrap_ci95_high": b,
                                       "bootstrap_valid_draws": n_boot})
    trends = pd.DataFrame(trend_rows)
    trends["hac_q_bh"] = np.nan
    for _, subset in trends.dropna(subset=["hac_p"]).groupby(["rule", "support"]):
        # Report one exploratory family across the four windows and three thresholds.
        trends.loc[subset.index, "hac_q_bh"] = multipletests(subset.hac_p, method="fdr_bh")[1]
    files = {
        "snowline_annual.csv": pd.DataFrame(annual_rows),
        "snowline_annual_trends.csv": trends,
        "snowline_annual_fits.csv": pd.DataFrame(prediction_rows),
        "snowline_period_summary.csv": pd.DataFrame(period_rows),
        "snowline_paired_shifts.csv": pd.DataFrame(shift_rows),
        "snowline_longitude_crossings.csv": pd.DataFrame(lon_rows),
    }
    for fname, frame in files.items():
        frame.to_csv(CSV / fname, index=False)
    trends.query("rule == 'main'").to_csv(AGENT / "annual_trends_main.csv", index=False)
    pd.DataFrame(shift_rows).query("rule == 'main' and longitude_block_deg == 10").to_csv(
        AGENT / "paired_shifts_main.csv", index=False)
    with (AGENT / "methods.json").open("w", encoding="utf-8") as file:
        json.dump({"thresholds": config["thresholds"], "rules_min_rook_cells_north_run_and_interior": RULES,
                   "patch_domain": "Full analysis domain; periodic longitude adjacency",
                   "column_rule": "Adjacent, valid, below-to-at-least-threshold south-to-north pair; latitude interpolated within grid pair only. Both endpoints and north run must lie in window. Keep equatorward crossing and record total candidates.",
                   "interior_rule": "Require the immediate west and east longitude pairs at the same crossing latitude bracket to be valid; removes crossings beside coastal or missing-cell quadrilaterals.",
                   "trend": "OLS on annual longitude medians, calendar-lag Bartlett HAC(3), small-sample t and 95% intervals; 8 complete response winters minimum. BH q values over 12 (region, threshold) comparisons for each rule and support, excluding undefined tests.",
                   "shift": "Median late minus early across same longitude columns; positive northward. Secondary difference of period medians on the same paired support. Spatial 5/10/20-degree moving-block interval conditional on the fixed decades, circular only for NH.",
                   "source_data": list(files)}, file, indent=2)
    print(trends.query("rule == 'main' and support == 'available'").to_string(index=False))
    print(pd.DataFrame(shift_rows).query("rule == 'main' and longitude_block_deg == 10").to_string(index=False))


if __name__ == "__main__":
    main()
