"""Check analysis invariants and audit the exact scientific figure PDFs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import xarray as xr

from figure_style import FIGURES, QA, ROOT, SKILL

OUT = ROOT / "output"
CSV = OUT / "csv"
LOG = ROOT / "logs"


def verify_data():
    data = xr.load_dataset(OUT / "processed/djf_winter_fields.nc", engine="h5netcdf")
    years = np.arange(1980, 2023)
    mask = data.analysis_mask.values.astype(bool)
    area = data.cell_area.values[mask]
    assert np.array_equal(data.winter_year.values, years)
    assert mask.sum() == 11844
    scf, swe, t2m = (data[name].values for name in
                     ("snow_cover_fraction", "snow_water_equivalent", "temperature_2m"))
    assert all(np.isfinite(field[:, mask]).all() and np.isnan(field[:, ~mask]).all()
               for field in (scf, swe, t2m))
    g = pd.read_csv(CSV / "global_metrics.csv")
    assert np.array_equal(g.winter_year, years)
    np.testing.assert_allclose(g.sca_km2, (scf[:, mask] @ area) / 1e6, rtol=1e-10)
    np.testing.assert_allclose(g.storage_a_km3, (swe[:, mask] @ area) / 1e12, rtol=1e-10)
    np.testing.assert_allclose(g.storage_b_km3, ((scf[:, mask] * swe[:, mask]) @ area) / 1e12, rtol=1e-10)
    np.testing.assert_allclose(g.swe_mm, (swe[:, mask] @ area) / area.sum(), rtol=1e-10)
    np.testing.assert_allclose(g.t2m_c, (t2m[:, mask] @ area) / area.sum(), rtol=1e-10)
    check = pd.read_csv(CSV / "backup_validation.csv")
    assert len(check) == 2 * 3 * len(years) and check.missing_mismatch.max() == 0
    lags = pd.read_csv(CSV / "temperature_lags.csv")
    assert len(lags) == 80 and np.all(lags.n == 43 - lags.lag.abs())
    assert np.all(lags.snow_first_year - lags.temp_first_year == lags.lag)
    assert np.all(lags.snow_last_year - lags.temp_last_year == lags.lag)
    response = pd.read_csv(CSV / "temperature_response.csv")
    np.testing.assert_allclose(response.detrended_r, response.partial_r_year)
    np.testing.assert_allclose(response.detrended_r ** 2, response.r2_detrended)
    annual = pd.read_csv(CSV / "snowline_annual_trends.csv")
    primary = annual.query("threshold == .50 and rule == 'main' and region == 'NH' and support == 'available'").iloc[0]
    assert primary.n_winters == 43 and primary.hac_ci95_low > 0
    step = pd.read_csv(CSV / "global_assimilation_step_sensitivity.csv")
    return {"winter_count": len(years), "fixed_cells": int(mask.sum()), "area_km2": float(area.sum()/1e6),
            "backup_missing_mismatches": int(check.missing_mismatch.sum()), "lag_tests": len(lags),
            "assimilation_step_rows": len(step), "boundary_slope_deg_decade": primary.slope_deg_per_decade}


def validate_figures():
    scripts = SKILL / "scripts"
    plots = sorted(FIGURES.glob("fig*.pdf"))
    assert len(plots) == 12
    assert {int(f.stem[3:5]) for f in plots} == set(range(1, 12))
    bundled = QA / "publication_figure_source.py"
    python_sources = [ROOT / "code/figure_style.py", *sorted((ROOT / "code").glob("08_*figure*.py"))]
    bundled.write_text("\n\n".join(f.read_text(encoding="utf-8") for f in python_sources),
                       encoding="utf-8")
    with (QA / "publication_source_preflight.json").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, str(scripts / "validate_figure.py"), str(bundled), "--json"],
                       check=True, stdout=stream)
    summary = {}
    for pdf in plots:
        assert all(pdf.with_suffix(ext).is_file() for ext in (".svg", ".png"))
        alignment = json.loads((QA / f"{pdf.stem}.alignment.json").read_text(encoding="utf-8-sig"))
        assert alignment["verdict"] in {"PASS", "NOT APPLICABLE"}, alignment["verdict"]
        with (QA / f"{pdf.stem}.text.json").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, str(scripts / "audit_pdf_text.py"), str(pdf),
                            "--min-pt", "5", "--json"], check=True, stdout=stream)
        collision = QA / f"{pdf.stem}.collision-audit.json"
        subprocess.run([sys.executable, str(scripts / "audit_figure_collisions.py"), str(pdf),
                        "--json-out", str(collision), "--strict"], check=True, stdout=subprocess.DEVNULL)
        summary[pdf.stem] = {"alignment": alignment["verdict"], "glyph_floor_pt": 5, "collision": "PASS"}
    return summary


def verify_report():
    pdf = OUT / "北半球冬季积雪变化研究报告.pdf"
    if not pdf.exists():
        return {"compiled_pdf": "absent"}
    with fitz.open(pdf) as doc:
        pages = [page.get_text() for page in doc]
    assert len(pages) > 5 and "参考文献" in pages[-1]
    assert not any("??" in page for page in pages)
    compile_log = next((LOG / n for n in ("latex_build.log",) if (LOG / n).is_file()), None)
    if compile_log:
        s = compile_log.read_text(encoding="utf-8-sig")
        assert not any(token in s for token in ("Overfull", "Undefined control sequence", "undefined references",
                                               "Missing character", "LaTeX Error"))
    return {"compiled_pdf": pdf.name, "pages": len(pages)}


def main():
    result = {"statistics": verify_data(), "figure_pdfs": validate_figures(), "report": verify_report()}
    (LOG / "delivery_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified numerical invariants, {len(result['figure_pdfs'])} figure PDFs and report.")


if __name__ == "__main__":
    main()
