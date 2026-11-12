from __future__ import annotations

import calendar
import hashlib
import importlib.metadata
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
LOG = ROOT / "logs"
CSV = OUT / "csv"
PROCESSED = OUT / "processed"
MASKS = OUT / "region_masks"
KEYS = {"scf": "snow_cover_fraction", "swe": "snow_water_equivalent",
        "t2m": "temperature_2m"}
WINDOWS = {"Europe": [35, 72, -15, 60], "Asia": [5, 85, 60, 180],
           "North America": [15, 85, -170, -50]}
BANDS = {"0–30": [0, 30], "30–50": [30, 50], "50–65": [50, 65], "65–90": [65, 91]}


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode), encoding="utf-8")


def load(z, basename):
    with xr.open_dataset(io.BytesIO(z.read("data/" + basename)), engine="h5netcdf") as ds:
        return ds.load()


def main(archive):
    for p in [LOG, CSV, PROCESSED, MASKS]:
        p.mkdir(parents=True, exist_ok=True)
    summary = {"archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": {}}
    rows = []
    with zipfile.ZipFile(archive) as z:
        summary["archive_inventory"] = [{"name": a.filename, "bytes": a.file_size, "crc": f"{a.CRC:08x}"} for a in z.infolist()]
        intro, month, backup = [load(z, f"snow_{tag}.nc") for tag in ("intro_1980", "monthly", "winter_backup")]
    for tag, ds in zip(("intro_1980", "monthly", "winter_backup"), (intro, month, backup)):
        path = f"snow_{tag}.nc"
        record = {"dimensions": dict(ds.sizes), "attributes": ds.attrs}
        domain = ds.analysis_mask.values == 1
        record["analysis_grid_cells"] = int(domain.sum())
        record["analysis_area_m2"] = ds.cell_area.values[domain].sum()
        record["coordinates"] = {}
        for key in ("latitude", "longitude", "winter_year", "time"):
            if key in ds:
                a = ds[key].values
                details = {"first": str(a[0]), "last": str(a[-1]),
                           "ascending": bool(np.all(a[1:] >= a[:-1]))}
                if key in ("latitude", "longitude"):
                    details["step_degrees"] = np.unique(np.diff(a)).tolist()
                record["coordinates"][key] = details
        for name, v in ds.variables.items():
            row = {"file": path, "file_bytes": next(a["bytes"] for a in summary["archive_inventory"] if a["name"] == "data/" + path),
                   "name": name, "dims": ",".join(v.dims), "shape": "×".join(map(str, v.shape)),
                   "dtype": str(v.dtype), "units": v.attrs.get("units", v.encoding.get("units", "")),
                   "calendar": v.encoding.get("calendar", ""), "long_name": v.attrs.get("long_name", ""),
                   "fill_value": v.encoding.get("_FillValue", v.attrs.get("_FillValue", "")),
                   "missing_value": v.encoding.get("missing_value", v.attrs.get("missing_value", "")),
                   "attributes": json.dumps(v.attrs, default=encode, ensure_ascii=False)}
            if np.issubdtype(v.dtype, np.number):
                a = v.values
                val = a[np.isfinite(a)]
                row["n_missing"] = a.size - val.size
                if val.size:
                    row.update(zip(("min", "p01", "p25", "median", "p75", "p99", "max"),
                                   np.quantile(val, [0, .01, .25, .5, .75, .99, 1])))
                if name.endswith("_mask"):
                    row["value_counts"] = str(dict(zip(*np.unique(a, return_counts=True))))
            rows.append(row)
        for label, space in [("inside", domain), ("outside", ~domain)]:
            a = ds[KEYS["swe"]].values[:, space]
            record["swe_" + label] = {"zeros": int((a == 0).sum()), "missing": int((~np.isfinite(a)).sum()),
                                      "ge_9999": int((a >= 9999).sum()), "max": np.nanmax(a)}
        summary["files"][path] = record
    pd.DataFrame(rows).to_csv(LOG / "metadata_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(CSV / "metadata_audit.csv", index=False)

    time = pd.DatetimeIndex(month.time.values)
    winter = time.year + (time.month == 12)
    assert np.array_equal(winter, month.winter_year.values)
    days = np.array([calendar.monthrange(t.year, t.month)[1] for t in time])
    assert np.array_equal(days, month.days_in_month.values)
    years = np.unique(winter)
    assert np.array_equal(years, backup.winter_year.values)
    for key in ("latitude", "longitude", "analysis_mask", "cell_area"):
        np.testing.assert_array_equal(month[key], backup[key])
    for year in years:
        assert time.month[winter == year].tolist() == [12, 1, 2]

    mask = month.analysis_mask.values == 1
    area = month.cell_area.values
    assert np.all(area > 0)
    assert np.array_equal(mask, (month.land_mask.values == 1) & (month.fixed_10m_mask.values == 0))
    assert month[KEYS["t2m"]].attrs["units"] == "degree_Celsius"
    assert month[KEYS["swe"]].attrs["units"] == "mm"
    assert month[KEYS["scf"]].attrs["units"] == "1"

    ds = xr.Dataset(coords={"winter_year": years, "latitude": month.latitude, "longitude": month.longitude},
                    attrs={"season": "DJF, labelled by January and February year",
                           "weighting": "Actual days per month; all 3 months finite; fixed analysis_mask applied",
                           "swe_spatial_support": "Unresolved by supplied metadata; report area and snow-area weighting scenarios",
                           "source_archive_sha256": summary["sha256"]})
    for key in ("analysis_mask", "land_mask", "fixed_10m_mask", "cell_area"):
        ds[key] = month[key]
    validation = []
    active_rows = []
    for short, name in KEYS.items():
        stored = month[name].values
        raw = stored.astype("float64")
        raw_djf = []
        legacy_djf = []
        for year in years:
            select = winter == year
            raw_djf.append(np.average(raw[select], axis=0, weights=days[select]))
            weighted32 = (stored[select] * days[select].astype("float32")[:, None, None]).sum(axis=0)
            legacy_djf.append(weighted32.astype("float64") / days[select].sum())
        raw_djf = np.stack(raw_djf)
        other = backup[name].transpose("winter_year", "latitude", "longitude").values
        np.testing.assert_array_equal(np.stack(legacy_djf), other)
        for scope, space in (("full_grid", np.ones_like(mask)), ("analysis_mask", mask)):
            for i, year in enumerate(years):
                left, right = raw_djf[i][space], other[i][space]
                both = np.isfinite(left) & np.isfinite(right)
                delta = left[both] - right[both]
                validation.append({"variable": name, "scope": scope, "winter_year": year, "paired_n": both.sum(),
                                   "missing_mismatch": (np.isfinite(left) ^ np.isfinite(right)).sum(),
                                   "max_abs": np.max(np.abs(delta)), "mae": np.mean(np.abs(delta)),
                                   "rmse": np.sqrt(np.mean(delta ** 2))})
        # Inspect validity at the monthly level before seasonal aggregation can dilute a fill value.
        if short == "swe":
            valid = (raw >= 0) & (raw < 9999)
        elif short == "scf":
            valid = (raw >= 0) & (raw <= 1)
        else:
            valid = np.isfinite(raw)
        valid = np.stack([valid[winter == year].all(axis=0) for year in years])
        field = np.where(valid & mask, raw_djf, np.nan)
        ds[name] = (("winter_year", "latitude", "longitude"), field, month[name].attrs)
        ds["complete_" + short] = (("latitude", "longitude"), np.isfinite(field).all(axis=0).astype("int8"))
        for i, year in enumerate(years):
            active = np.isfinite(field[i])
            active_rows.append({"winter_year": year, "variable": name, "valid_n": active.sum(),
                                "valid_area_km2": area[active].sum() / 1e6})
    validation = pd.DataFrame(validation)
    validation.to_csv(CSV / "backup_validation.csv", index=False)
    pd.DataFrame(active_rows).to_csv(CSV / "valid_coverage.csv", index=False)
    summary["backup"] = validation.groupby(["scope", "variable"])[["missing_mismatch", "max_abs", "mae", "rmse"]].max().reset_index().to_dict(orient="records")
    summary["backup_precision_check"] = "Reproduced exactly using float32 multiplication and summation followed by division by integer DJF days. Analysis uses float64 arithmetic."
    dump(LOG / "metadata_audit.json", summary)
    assert validation.missing_mismatch.max() == 0
    assert np.isfinite(ds[KEYS["scf"]].values[:, mask]).all()

    y, x = np.meshgrid(month.latitude.values, month.longitude.values, indexing="ij")
    region = {name: mask & (y >= a) & (y <= b) & (x >= c) & (x < d)
              for name, (a, b, c, d) in WINDOWS.items()}
    assert np.max(np.sum(list(region.values()), axis=0)) == 1
    bands = {name: mask & (y >= a) & (y < b) for name, (a, b) in BANDS.items()}
    assert np.array_equal(np.sum(list(bands.values()), axis=0), mask)
    np.savez_compressed(MASKS / "fixed_masks.npz", NH=mask, **region, **bands)
    dump(OUT / "analysis_config.yaml", {"variables": KEYS, "windows": WINDOWS, "latitude_bands": BANDS,
                                        "window_rule": "inclusive latitude, half-open longitude",
                                        "band_rule": "half-open (final band includes 90 N)",
                                        "analysis_years": years, "early": [1980, 1989], "late": [2013, 2022],
                                        "thresholds": [.25, .5, .75], "bootstrap_seed": 20261111,
                                        "bootstrap_draws": 3999, "bootstrap_blocks": [3, 5, 7],
                                        "hac_maxlags": 3, "alpha": .05})
    enc = {name: {"zlib": True, "complevel": 4} for name in KEYS.values()}
    ds.to_netcdf(PROCESSED / "djf_winter_fields.nc", engine="h5netcdf", encoding=enc)
    dump(LOG / "versions.json", {"python": sys.version, **{name: importlib.metadata.version(name) for name in
         ["numpy", "pandas", "xarray", "h5netcdf", "h5py", "scipy", "statsmodels", "matplotlib"]}})
    print(f"{len(years)} complete DJF calendar groups; {mask.sum()} analysis cells; {area[mask].sum()/1e6:.3f} km²")
    print(validation.groupby(["scope", "variable"])[["max_abs", "missing_mismatch"]].max().to_string())
    print(pd.DataFrame(active_rows).groupby("variable").valid_n.agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "data.zip")
