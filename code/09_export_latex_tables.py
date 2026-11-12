"""Typeset verified source-data estimates without manual transcription."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "output" / "csv"
TABLES = ROOT / "output" / "tables"
GROUPS = {"NH": "全分析域", "Europe": "欧洲窗口", "Asia": "亚洲窗口",
          "North America": "北美洲窗口", "0–30": "$0$--$30^\circ$N",
          "30–50": "$30$--$50^\circ$N", "50–65": "$50$--$65^\circ$N",
          "65–90": "$65$--$90^\circ$N"}


def prob(p):
    if p < .001:
        mantissa, power = f"{p:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(power)}}}"
    return f"{p:.3f}"


def interval(beta, lo, hi, digits=3):
    return f"${beta:.{digits}f}\\;[{lo:.{digits}f},\\ {hi:.{digits}f}]$"


def stacked_interval(beta, lo, hi):
    return rf"\shortstack[r]{{ $ {beta:.3f} $\\$[{lo:.3f},\ {hi:.3f}]$ }}"


def tabular(name, cols, headings, rows):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(CSV / "trend_global.csv").query("coverage == 'fixed_mask'").set_index("metric")
    metrics = {"sca_km2": r"SCA ($10^6$ km$^2$)", "scr_pct": "SCR (pp)",
               "swe_mm": "平均 SWE (mm)", "t2m_c": r"T2m ($^\circ$C)",
               "storage_a_km3": r"$V_A$ (km$^3$)", "storage_b_km3": r"$V_B$ (km$^3$)"}
    rows = []
    for metric, label in metrics.items():
        a = data.loc[metric]
        factor = 1e6 if metric == "sca_km2" else 1
        digits = 1 if metric.startswith("storage") else 3
        rows.append([label, interval(a.slope/factor, a.ci_low/factor, a.ci_high/factor, digits),
                     f"${prob(a.p_hac)}$", f"${a.r2:.3f}$", f"${a.n_eff:.1f}$"])
    tabular("table_global", "lrrrr", ["指标", r"斜率 [95\% CI]", r"$p_{\rm HAC}$", r"$R^2$", r"$n_{\mathrm{eff}}$"], rows)

    data = pd.read_csv(CSV / "regional_latitude_trends.csv").set_index(["group", "metric"])
    rows = []
    for group, label in list(GROUPS.items())[1:]:
        part = [data.loc[(group, metric)] for metric in ("scf", "swe", "t2m")]
        rows.append([label, f"{part[0].mean_area_km2/1e6:.2f}"] +
                    [stacked_interval(a.slope, a.ci_low, a.ci_high) for a in part])
    tabular("table_spatial", "lrrrr", ["分组", r"面积 ($10^6$ km$^2$)",
            "SCF (pp)", "SWE (mm)", r"T2m ($^\circ$C)"], rows)

    data = pd.read_csv(CSV / "snowline_paired_shifts.csv").query("rule == 'main' and longitude_block_deg == 10")
    rows = []
    for a in data.itertuples():
        rows.append([f"{a.threshold:.2f}", GROUPS[a.region], a.paired_n_longitudes,
                     interval(a.paired_median_shift_deg, a.spatial_bootstrap_ci95_low, a.spatial_bootstrap_ci95_high, 3)])
    tabular("table_boundary", "llrr", ["SCF 阈值", "范围", "配对经度数", r"纬度位移 ($^\circ$) [95\% CI]"], rows)

if __name__ == "__main__":
    main()
