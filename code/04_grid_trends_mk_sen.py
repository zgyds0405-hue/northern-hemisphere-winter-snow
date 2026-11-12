"""Calculate comparable grid trends, multiplicity and spatial clustering."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CSV = OUT / "csv"
AGENT = OUT / "agent_trend"
CONFIG = json.loads((OUT / "analysis_config.yaml").read_text(encoding="utf-8"))

UNITS = {"scf": "percentage points", "swe": "mm", "t2m": "degC"}
MEAN_UNITS = {"scf": "fraction", "swe": "mm", "t2m": "degC"}


def save_table(table, name):
    for directory in (CSV, AGENT):
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / name, index=False)


def grid_regressions(year, values, lags):
    """Vectorized temporal inference for cells with complete winter coverage."""
    n, width = values.shape
    a, b = np.triu_indices(n, 1)
    x = (year - year.mean()) / 10
    sxx = np.dot(x, x)
    centered = values - values.mean(axis=0)
    beta = x @ centered / sxx
    constant = np.ptp(values, axis=0) == 0
    beta[constant] = 0
    residual = centered - x[:, None] * beta
    residual[:, constant] = 0
    score = residual * x[:, None]
    hac_meat = np.einsum("ij,ij->j", score, score)
    for lag in range(1, lags + 1):
        hac_meat += 2 * (1 - lag / (lags + 1)) * np.einsum("ij,ij->j", score[lag:], score[:-lag])
    sse = np.einsum("ij,ij->j", residual, residual)
    sst = np.einsum("ij,ij->j", centered, centered)
    r2 = 1 - np.divide(sse, sst, out=np.full(width, np.nan), where=~constant)
    prev, next_ = residual[:-1], residual[1:]
    prev, next_ = prev - prev.mean(axis=0), next_ - next_.mean(axis=0)
    lag_norm = np.linalg.norm(prev, axis=0) * np.linalg.norm(next_, axis=0)
    rho = np.divide(np.einsum("ij,ij->j", prev, next_), lag_norm,
                    out=np.full(width, np.nan), where=lag_norm > 0)
    rho_pos = np.maximum(np.nan_to_num(rho), 0)
    n_eff = n * (1 - rho_pos) / (1 + rho_pos)

    differences = values[b] - values[a]
    mk_s = np.sign(differences).sum(axis=0)
    mk_var = np.empty(width)
    for i, column in enumerate(values.T):
        _, ties = np.unique(column, return_counts=True)
        mk_var[i] = (n * (n - 1) * (2 * n + 5) - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18
    mk_z = np.divide(mk_s - np.sign(mk_s), np.sqrt(mk_var), out=np.zeros(width), where=mk_var > 0)
    sen_slope = np.median(differences / ((year[b] - year[a]) / 10)[:, None], axis=0)
    result = dict(n=np.repeat(n, width), n_eff=n_eff, r2=r2, rho1=rho, slope=beta,
                  mk_s=mk_s.astype(int), mk_z=mk_z, p_mk=2 * stats.norm.sf(np.abs(mk_z)),
                  sen_slope=sen_slope)
    critical = stats.t.ppf(.975, n - 2)
    for method, se in [("ols", np.sqrt(sse / (n - 2) / sxx)),
                       ("hac", np.sqrt(np.maximum(0, hac_meat) * n / (n - 2)) / sxx)]:
        t_stat = np.divide(np.abs(beta), se, out=np.where(beta == 0, 0.0, np.inf), where=se > 0)
        result[f"p_{method}"] = 2 * stats.t.sf(t_stat, n - 2)
        result[f"ci_low_{method}"] = beta - critical * se
        result[f"ci_high_{method}"] = beta + critical * se
    return pd.DataFrame(result)


def spatial_moran(field, mask, seed, draws=999):
    """Random-label test of trend-map similarity with undirected rook adjacency."""
    coord_ids = np.full(mask.shape, -1, dtype=int)
    coord_ids[mask] = np.arange(mask.sum())
    lon_neighbors = np.roll(coord_ids, -1, axis=1)
    along_lon = (coord_ids >= 0) & (lon_neighbors >= 0)
    along_lat = (coord_ids[:-1] >= 0) & (coord_ids[1:] >= 0)
    edge_from = np.r_[coord_ids[along_lon], coord_ids[:-1][along_lat]]
    edge_to = np.r_[lon_neighbors[along_lon], coord_ids[1:][along_lat]]
    x = field[mask]
    x = x - x.mean()
    norm = np.dot(x, x)
    scale = x.size / edge_from.size / norm

    def moran(z):
        return scale * np.sum(z[edge_from] * z[edge_to])

    observed = moran(x)
    rng = np.random.default_rng(seed)
    null = np.array([moran(rng.permutation(x)) for _ in range(draws)])
    return dict(moran_i=observed, moran_expected=-1 / (len(x) - 1),
                moran_p_high=(1 + np.count_nonzero(null >= observed)) / (draws + 1),
                moran_null_95=np.quantile(null, .95), moran_permutations=draws,
                rook_edges=edge_from.size)


def summarize_grid(table, field, mask, seed):
    area = table.area_km2.to_numpy()
    beta = table.slope.to_numpy()
    fraction = lambda select: area[select].sum() / area.sum()
    significant = table.q_hac.to_numpy() < CONFIG["alpha"]
    return {
        "tested_n": len(table), "tested_area_km2": area.sum(),
        "grid_positive_n": np.sum(beta > 0), "grid_negative_n": np.sum(beta < 0),
        "grid_zero_n": np.sum(beta == 0),
        "positive_area_fraction": fraction(beta > 0),
        "negative_area_fraction": fraction(beta < 0),
        "zero_area_fraction": fraction(beta == 0),
        "hac_bh_significant_n": significant.sum(), "hac_bh_significant_grid_fraction": significant.mean(),
        "hac_bh_significant_area_fraction": fraction(significant),
        "hac_bh_positive_area_fraction": fraction(significant & (beta > 0)),
        "hac_bh_negative_area_fraction": fraction(significant & (beta < 0)),
        "ols_bh_significant_grid_fraction": (table.q_ols < CONFIG["alpha"]).mean(),
        "mk_bh_significant_grid_fraction": (table.q_mk < CONFIG["alpha"]).mean(),
        "mean_climatology_area_weighted": np.average(table.climatology, weights=area),
        "p05_climatology_grid_unweighted": np.quantile(table.climatology, .05),
        "p95_climatology_grid_unweighted": np.quantile(table.climatology, .95),
        "min_climatology": table.climatology.min(), "max_climatology": table.climatology.max(),
        **spatial_moran(field, mask, seed),
    }


def main():
    path = OUT / "processed"
    with xr.open_dataset(path / "djf_winter_fields.nc", engine="h5netcdf") as source:
        data = source.load()
    year = data.winter_year.to_numpy()
    mask = data.analysis_mask.to_numpy().astype(bool)
    area = data.cell_area.to_numpy() / 1e6
    lat, lon = np.meshgrid(data.latitude.to_numpy(), data.longitude.to_numpy(), indexing="ij")
    result = data[["analysis_mask", "cell_area"]].copy()
    dims = ("latitude", "longitude")
    tables, summaries = [], []
    betas, significant = {}, {}
    lags = CONFIG["hac_maxlags"]
    for variable, name in CONFIG["variables"].items():
        raw = data[name].to_numpy()
        complete = mask & np.isfinite(raw).all(axis=0)
        values = raw[:, complete]
        factor = 100 if variable == "scf" else 1
        blocks = [grid_regressions(year, values[:, start:start + 1024] * factor, lags)
                  for start in range(0, values.shape[1], 1024)]
        estimates = pd.concat(blocks, ignore_index=True)
        for method in ("hac", "ols", "mk"):
            estimates[f"q_{method}"] = multipletests(estimates[f"p_{method}"], method="fdr_bh")[1]
        climatology = values.mean(axis=0)
        metadata = pd.DataFrame({"variable": variable, "latitude": lat[complete], "longitude": lon[complete],
                                 "area_km2": area[complete], "climatology": climatology,
                                 "climatology_units": MEAN_UNITS[variable],
                                 "trend_units_per_decade": UNITS[variable]})
        tables.append(pd.concat([metadata, estimates], axis=1))
        for field, vector, units in [
            ("mean", climatology, MEAN_UNITS[variable]),
            ("slope", estimates.slope, UNITS[variable] + " per decade"),
            ("q_hac", estimates.q_hac, "1"),
        ]:
            raster = np.full(mask.shape, np.nan)
            raster[complete] = vector
            result[f"{variable}_{field}"] = (dims, raster, {"units": units})
        beta = result[f"{variable}_slope"].to_numpy()
        betas[variable] = beta
        significant[variable] = result[f"{variable}_q_hac"].to_numpy() < CONFIG["alpha"]
        summaries.append({"variable": variable, "mask_n": mask.sum(),
                          "complete_mask_n": complete.sum(), "n_winters": year.size,
                          **summarize_grid(tables[-1], beta, complete, CONFIG["bootstrap_seed"])})
    result.attrs.update({"season": "December–February 1980–2022",
                         "trend": "OLS slopes per decade; HAC lag 3; Benjamini–Hochberg per variable"})
    result.to_netcdf(path / "grid_trends.nc", engine="h5netcdf", encoding={
        name: {"zlib": True, "complevel": 4} for name in result.data_vars
    })
    all_trends = pd.concat(tables, ignore_index=True)
    all_trends["slope_fraction_per_decade"] = (all_trends.slope / 100).where(all_trends.variable == "scf")
    save_table(all_trends, "trend_grid.csv")
    summary = pd.DataFrame(summaries)
    save_table(summary, "grid_summary.csv")
    paired_rows = []
    for first, second in combinations(betas, 2):
        left, right = betas[first], betas[second]
        paired = np.isfinite(left) & np.isfinite(right)
        denominator = area[paired].sum()
        for s1, s2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            category = paired & (np.sign(left) == s1) & (np.sign(right) == s2)
            both_sig = category & significant[first] & significant[second]
            paired_rows.append({"variable_1": first, "variable_2": second,
                                "sign_1": s1, "sign_2": s2, "paired_area_km2": denominator,
                                "area_fraction": area[category].sum() / denominator,
                                "both_hac_bh_area_fraction": area[both_sig].sum() / denominator})
    save_table(pd.DataFrame(paired_rows), "grid_trend_overlap.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
