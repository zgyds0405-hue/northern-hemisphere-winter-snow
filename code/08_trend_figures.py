"""Draw the climatology, domain time series and grid-trend evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import xarray as xr

from figure_style import FIG_WIDTH, INK, NEUTRAL, ROOT, SCF, SWE, T2M, map_axes, map_field, panel, save, setup

OUT = ROOT / "output"

TREND_MAP = {
    "scf": ("SCF (pp per decade)", 6, [-6, -3, 0, 3, 6], "min"),
    "swe": ("SWE (mm per decade)", 20, [-20, -10, 0, 10, 20], "both"),
    "t2m": ("T2m (°C per decade)", 1, [-1, -.5, 0, .5, 1], "max"),
}
CLIMATOLOGY_MAP = {
    "scf": ("SCF (fraction)", "Blues", 0, 1, [0, .25, .5, .75, 1], "neither"),
    "swe": ("SWE (mm)", "GnBu", 0, 200, [0, 50, 100, 150, 200], "max"),
    "t2m": ("T2m (°C)", "RdBu_r", -40, 30, [-40, -20, 0, 15, 30], "both"),
}
TREND_CMAP = "RdBu_r"


def trend_map(ax, grid, key, *, stipple=True):
    """Use the same trend-map geometry and scales in Figs. 4 and 10."""
    label, limit, ticks, extend = TREND_MAP[key]
    lon, lat = grid.longitude.values, grid.latitude.values
    field = map_field(ax, lon, lat, grid[f"{key}_slope"].values,
                      cmap=TREND_CMAP, vmin=-limit, vmax=limit)
    if stipple:
        rows, cols = np.where(grid[f"{key}_q_hac"].values < .05)
        ax.scatter(lon[cols], lat[rows], s=.23, c=INK, alpha=.50, marker="o",
                   linewidths=0, rasterized=True, zorder=3)
    return field, label, ticks, extend


def map_figure(grid, kind):
    """Three comparable full-domain layers retain identical map geometry."""
    fig = plt.figure(figsize=(FIG_WIDTH, 5.88))
    frame = fig.add_gridspec(3, 2, left=.113, right=.884, bottom=.080, top=.902,
                            height_ratios=[1, 1, 1], width_ratios=[30, 1],
                            hspace=.28, wspace=.065)
    axes = []
    for i, (key, letter) in enumerate(zip(TREND_MAP, "abc")):
        ax = fig.add_subplot(frame[i, 0])
        cax = fig.add_subplot(frame[i, 1])
        axes.append(ax)
        if kind == "trend":
            image, label, ticks, extend = trend_map(ax, grid, key)
        else:
            label, cmap, lo, hi, ticks, extend = CLIMATOLOGY_MAP[key]
            scale = dict(norm=TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)) if key == "t2m" else dict(vmin=lo, vmax=hi)
            image = map_field(ax, grid.longitude.values, grid.latitude.values,
                              grid[f"{key}_mean"].values, cmap=cmap, **scale)
        map_axes(ax, lon_labels=i == 2)
        panel(ax, letter, x=-.115, y=1.015)
        color = fig.colorbar(image, cax=cax, ticks=ticks, extend=extend)
        color.outline.set_linewidth(.45)
        color.ax.tick_params(labelsize=6.5, length=2)
        color.set_label(label, fontsize=7.2)
    heading = "Winter mean fields" if kind == "mean" else "Linear trends in winter fields"
    fig.text(.113, .958, heading, fontsize=9.3, fontweight="bold")
    fig.text(.113, .928, "Fixed Northern Hemisphere analysis domain  ·  DJF 1980–2022  ·  43 winters",
             fontsize=6.9, color=NEUTRAL)
    if kind == "trend":
        fig.legend(handles=[Line2D([], [], marker=".", color=INK, markersize=4, linestyle="none",
                                   label="HAC–BH q < 0.05")],
                   loc="lower right", bbox_to_anchor=(.899, .005), borderaxespad=0)
    save(fig, "fig02_climatology" if kind == "mean" else "fig04_grid_trends", axes=axes)


def trend_line(ax, year, value, color, *, label=None, linestyle="-", marker="o", scale=1):
    """Pointwise 95% confidence band for the OLS fitted mean using HAC covariance."""
    value = value / scale
    design = sm.add_constant((year - year.mean()) / 10)
    result = sm.OLS(value, design).fit().get_robustcov_results(cov_type="HAC", maxlags=3,
                                                             use_correction=True, use_t=True)
    mean = design @ result.params
    spread = np.sqrt(np.einsum("ij,jk,ik->i", design, result.cov_params(), design))
    spread *= stats.t.ppf(.975, len(year) - design.shape[1])
    ax.fill_between(year, mean - spread, mean + spread, color=color, alpha=.11, linewidth=0)
    ax.plot(year, value, color=color, lw=.66, alpha=.76, marker=marker, ms=1.85,
            mew=0, label=label)
    ax.plot(year, mean, color=color, lw=1.12, linestyle=linestyle)


def series_figure(data):
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.22))
    fig.subplots_adjust(left=.106, right=.967, bottom=.183, top=.837, hspace=.54, wspace=.39)
    panels = [
        ("Snow-covered area (million km²)", (("sca_km2", SCF, None, "-", "o", 1e6),)),
        ("Area-weighted SWE (mm)", (("swe_mm", SWE, None, "-", "o", 1),)),
        ("Scenario storage (km³)", (("storage_a_km3", SWE, "A · grid-average", "-", "o", 1),
                                    ("storage_b_km3", SCF, "B · snow-area mean", "--", "s", 1))),
        ("Area-weighted T2m (°C)", (("t2m_c", T2M, None, "-", "o", 1),)),
    ]
    year = data.winter_year.to_numpy()
    for i, (ax, (title, series)) in enumerate(zip(axs.flat, panels)):
        panel(ax, "abcd"[i], x=-.166, y=1.07)
        ax.set_title(title, pad=5)
        ax.set_xlim(1979, 2023)
        ax.set_xticks([1980, 1990, 2000, 2010, 2020])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#e7ebef", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(labelbottom=i >= 2)
        for metric, color, label, line, mark, scale in series:
            trend_line(ax, year, data[metric].to_numpy(), color, label=label,
                       linestyle=line, marker=mark, scale=scale)
        if i >= 2:
            ax.set_xlabel("Winter year", labelpad=4)
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.106, .024),
               ncol=2, borderaxespad=0, handlelength=1.4)
    fig.text(.106, .952, "Snow extent and water storage decline during winter warming", fontsize=9.3, fontweight="bold")
    fig.text(.106, .902, "Fixed Northern Hemisphere analysis domain  ·  DJF 1980–2022  ·  43 winters",
             fontsize=6.9, color=NEUTRAL)
    save(fig, "fig03_global_series", axes=list(axs.flat))


def main():
    setup()
    with xr.open_dataset(OUT / "processed" / "grid_trends.nc", engine="h5netcdf") as grid:
        map_figure(grid, "mean")
        map_figure(grid, "trend")
    series_figure(pd.read_csv(OUT / "csv" / "global_metrics.csv"))


if __name__ == "__main__":
    main()
