"""Area-weighted summaries, regional trend contrasts and window sensitivities."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from spatial_stats import (
    BANDS, BLOCKS, CONFIG, GROUPS, METRICS, OUTPUT, REGIONS, boot_interval,
    decadal_time, fdr, paired_slope_boot, residualize, trend, write_table, zscore,
)


def area_summary(fields, area, mask):
    weights = area[mask]
    scf, swe, temp = [field[:, mask] for field in fields]
    data = {}
    for key, values in zip(["scf", "swe", "t2m"], (scf, swe, temp)):
        valid = np.isfinite(values)
        data[f"{key}_valid_grid_n"] = valid.sum(axis=1)
        data[f"{key}_valid_area_km2"] = (valid @ weights) / 1e6
        average = np.nansum(values * weights, axis=1) / (valid @ weights)
        data[key] = average * 100 if key == "scf" else average
    both = np.isfinite(scf) & np.isfinite(swe)
    data["volume_b_valid_area_km2"] = (both @ weights) / 1e6
    data["volume_b_valid_grid_n"] = both.sum(axis=1)
    data["sca"] = np.nansum(scf * weights, axis=1) / 1e6
    data["volume_a"] = np.nansum(swe * weights, axis=1) / 1e12
    data["volume_b"] = np.nansum(scf * swe * weights, axis=1) / 1e12
    data["analysis_area_km2"] = weights.sum() / 1e6
    data["analysis_grid_n"] = mask.sum()
    return pd.DataFrame(data)


def window_variants(ds):
    lon, lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
    domain = ds.analysis_mask.values == 1
    base = CONFIG["windows"]
    versions = {}
    for variant in ("contracted", "expanded", "shift_west", "shift_east"):
        windows = {}
        for key, (south, north, west, east) in base.items():
            if variant in ("contracted", "expanded"):
                sign = 1 if variant == "contracted" else -1
                south, north = south + sign * 4, north - sign * 4
                west, east = west + sign * 7, east - sign * 7
                if variant == "expanded":
                    # Keep Europe and Asia adjacent at 60 E in this scenario.
                    east = 60 if key == "Europe" else east
                    west = 60 if key == "Asia" else west
            else:
                sign = -1 if variant == "shift_west" else 1
                south, north = south + sign * 4, north + sign * 4
                west, east = west + sign * 7, east + sign * 7
            west, east = max(west, -180), min(east, 180)
            mask = domain & (lat >= south) & (lat <= north) & (lon >= west) & (lon < east)
            windows[key] = ((south, north, west, east), mask)
        assert np.max(sum(mask.astype(int) for _, mask in windows.values())) == 1
        versions[variant] = windows
    return versions


def series_table(ds, fields, masks):
    frames = []
    for family, keys in GROUPS.items():
        for group in keys:
            df = area_summary(fields, ds.cell_area.values, masks[group])
            df.insert(0, "winter_year", ds.winter_year.values)
            df.insert(0, "group", group)
            df.insert(0, "group_type", family)
            for metric in ("swe", "volume_a", "volume_b"):
                df[f"d_sca_{metric}"] = zscore(df.sca) - zscore(df[metric])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_series(series):
    slopes, bootstrap, decouple, decouple_boot = [], [], [], []
    for (kind, group), subset in series.groupby(["group_type", "group"], sort=False):
        years = subset.winter_year.to_numpy()
        all_values = subset[list(METRICS)].to_numpy()
        decoupling_metrics = ("swe", "volume_a", "volume_b")
        for i, (metric, unit) in enumerate(METRICS.items()):
            values = all_values[:, i]
            slopes.append({"group_type": kind, "group": group, "metric": metric, "unit_per_decade": unit,
                           "mean": values.mean(), "mean_area_km2": subset.analysis_area_km2.mean(),
                           "grid_n": subset.analysis_grid_n.iloc[0], **trend(years, values)})
        for length in BLOCKS:
            reps = paired_slope_boot(years, all_values, length)
            for i, metric in enumerate(METRICS):
                estimate = stats.linregress(decadal_time(years), all_values[:, i]).slope
                bootstrap.append({"group_type": kind, "group": group, "metric": metric,
                                  "block": length, "slope": estimate, **boot_interval(reps[:, i], estimate)})
        for metric in decoupling_metrics:
            raw = subset[["sca", metric]].to_numpy()
            if np.any(np.std(raw, axis=0) == 0):
                continue
            anomaly, _ = residualize(years, raw)
            z = zscore(raw)
            difference = z[:, 0] - z[:, 1]
            base = {"group_type": kind, "group": group, "swe_metric": metric, "n": years.size}
            ds = trend(years, difference)
            decouple.append({**base, "raw_r": stats.pearsonr(*raw.T).statistic,
                             "detrended_r": stats.pearsonr(*anomaly.T).statistic,
                             "z_sca_slope": stats.linregress(decadal_time(years), z[:, 0]).slope,
                             "z_swe_slope": stats.linregress(decadal_time(years), z[:, 1]).slope,
                             **ds})
            for length in BLOCKS:
                reps = paired_slope_boot(years, z, length)
                decouple_boot.append({**base, "block": length, "slope": ds["slope"],
                                      **boot_interval(reps[:, 0] - reps[:, 1], ds["slope"])})
    slope_df = fdr(pd.DataFrame(slopes), p="p_hac", family=["group_type", "metric"], column="q_hac")
    boot_df = pd.DataFrame(bootstrap)
    dc_df = fdr(pd.DataFrame(decouple), p="p_hac", family=["group_type", "swe_metric"], column="q_hac")
    dc_boot_df = fdr(pd.DataFrame(decouple_boot), p="p_boot", family=["group_type", "swe_metric", "block"], column="q_boot")
    return slope_df, boot_df, dc_df, dc_boot_df


def contrasts(series):
    records = []
    all_metrics = [*METRICS, "d_sca_swe", "d_sca_volume_a", "d_sca_volume_b"]
    for kind, members in (("window", REGIONS), ("latitude", BANDS)):
        subset = series[series.group_type == kind]
        for metric in all_metrics:
            pivot = subset.pivot(index="winter_year", columns="group", values=metric).reindex(columns=members)
            years, values = pivot.index.to_numpy(), pivot.to_numpy()
            if not np.isfinite(values).all():
                continue
            for length in BLOCKS:
                reps = paired_slope_boot(years, values, length)
                for i, j in combinations(range(len(members)), 2):
                    direct = trend(years, values[:, i] - values[:, j])
                    records.append({"group_type": kind, "metric": metric, "left": members[i], "right": members[j],
                                    "block": length, "difference": direct["slope"], "n": len(years),
                                    "p_hac": direct["p_hac"], "hac_low": direct["ci_low"], "hac_high": direct["ci_high"],
                                    **boot_interval(reps[:, i] - reps[:, j], direct["slope"])})
    result = pd.DataFrame(records)
    fdr(result, p="p_hac", family=["group_type", "metric", "block"], column="q_hac")
    return fdr(result, p="p_boot", family=["group_type", "metric", "block"], column="q_boot")


def window_sensitivity(ds, fields, masks, trends):
    years = ds.winter_year.values
    versions = window_variants(ds)
    versions["primary"] = {g: (CONFIG["windows"][g], masks[g]) for g in REGIONS}
    records = []
    for variant, windows in versions.items():
        for group, (bounds, mask) in windows.items():
            series = area_summary(fields, ds.cell_area.values, mask)
            series["d_sca_swe"] = zscore(series.sca) - zscore(series.swe)
            for metric in [*METRICS, "d_sca_swe"]:
                base = {"variant": variant, "group": group, "metric": metric,
                        "lat_min": bounds[0], "lat_max": bounds[1], "lon_min": bounds[2], "lon_max": bounds[3],
                        "area_km2": series.analysis_area_km2.iloc[0], "grid_n": mask.sum()}
                result = trend(years, series[metric])
                records.append({**base, **result})
    data = pd.DataFrame(records)
    data["rank_largest_slope_first"] = data.groupby(["variant", "metric"]).slope.rank(ascending=False, method="min").astype(int)
    primary = data[data.variant == "primary"].set_index(["group", "metric"])
    data["primary_slope"] = [primary.loc[(row.group, row.metric), "slope"] for row in data.itertuples()]
    data["primary_rank"] = [primary.loc[(row.group, row.metric), "rank_largest_slope_first"] for row in data.itertuples()]
    data["sign_stable"] = np.sign(data.slope) == np.sign(data.primary_slope)
    data["rank_stable"] = data.rank_largest_slope_first == data.primary_rank
    for row in trends[trends.group_type == "window"].itertuples():
        np.testing.assert_allclose(row.slope, primary.loc[(row.group, row.metric), "slope"])
    return data


def main():
    with xr.open_dataset(OUTPUT / "processed/djf_winter_fields.nc", engine="h5netcdf") as ds:
        fields = [ds[name].values for name in CONFIG["variables"].values()]
        with np.load(OUTPUT / "region_masks/fixed_masks.npz") as source:
            masks = {key: source[key].astype(bool) for key in source.files}
        series = series_table(ds, fields, masks)
        trends, boot, decoupling, decouple_boot = summarize_series(series)
        sensitivity = window_sensitivity(ds, fields, masks, trends)
    for data, stem in (
        (series, "regional_latitude_metrics"), (trends, "regional_latitude_trends"),
        (boot, "regional_latitude_block_bootstrap"), (decoupling, "area_storage_decoupling"),
        (decouple_boot, "area_storage_decoupling_bootstrap"), (contrasts(series), "regional_latitude_trend_contrasts"),
        (sensitivity, "approximate_geographic_windows_sensitivity"),
    ):
        write_table(data, stem, "spatial")
    print(trends[trends.metric.isin(["scf", "swe", "t2m"])][
        ["group", "metric", "slope", "ci_low", "ci_high", "p_hac", "q_hac", "n", "grid_n"]
    ].to_string(index=False))
    print(decoupling[decoupling.swe_metric == "swe"][
        ["group", "raw_r", "detrended_r", "slope", "ci_low", "ci_high", "p_hac"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
