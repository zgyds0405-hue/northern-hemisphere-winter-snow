"""Figures 5, 6 and 9 from the regional, latitudinal and temperature CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from figure_style import (
    FIG_WIDTH, INK, NEUTRAL, SCF, SWE, T2M, WINDOW_COLORS, panel, save, setup,
)
from spatial_stats import BANDS, CSV, REGIONS

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


VARIABLES = {"scf": ("SCF (%)", "SCF (pp per decade)", SCF),
             "swe": ("SWE (mm)", "SWE (mm per decade)", SWE),
             "t2m": ("T2m (°C)", "T2m (°C per decade)", T2M)}
WINDOW_SHORT = ["Europe", "Asia", "N. America"]
REGION_MARKERS = ["o", "s", "^"]
LAT_LABELS = [band + "° N" for band in BANDS]
ZERO = "#a0aab4"
BALANCED = LinearSegmentedColormap.from_list("temp_association", ["#1d557c", "#fbfbfa", "#ac5235"])


def read(stem):
    return pd.read_csv(CSV / (stem + ".csv"))


def axes_grid(rows, height, *, left=.14, right=.97, bottom=.10, top=.91, wspace=.39, hspace=.68):
    fig, axes = plt.subplots(rows, 3, figsize=(FIG_WIDTH, height / 25.4))
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    for letter, ax in zip("abcdefghi", axes.flat):
        panel(ax, letter, x=-.13, y=1.04)
    return fig, axes


def annual(ax, subset, metric, col, mark):
    years, vals = subset.winter_year.to_numpy(), subset[metric].to_numpy()
    fit = stats.linregress(years, vals)
    ax.plot(years, vals, lw=.52, marker=mark, ms=1.8, color=col, alpha=.63, markeredgewidth=0)
    ax.plot(years, fit.intercept + fit.slope * years, lw=1.0, color=col, ls=(0, (3, 2)))
    ax.set_xticks([1980, 2000, 2020])
    ax.set_xlim(1979, 2023)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def forest_mark(ax, x, low, high, row, *, color=INK, marker="o", fill=True):
    ax.hlines(row, low, high, color=color, lw=.95)
    ax.plot(x, row, linestyle="none", marker=marker, ms=4,
            color=color, markeredgewidth=.7, markerfacecolor=color if fill else "white")


def forest_scale(ax, labels, *, show_labels, xlabel, xbins=4):
    ax.set_ylim(len(labels) - .52, -.5)
    ax.set_yticks(range(len(labels)), labels if show_labels else [""] * len(labels))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.axvline(0, color=ZERO, lw=.65, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.margins(x=.16)


def window_fig(series, trends, dc, boot):
    fig, axs = axes_grid(3, 148, left=.15, top=.855, bottom=.09, hspace=.70)
    for j, (m, (title, slope, _)) in enumerate(VARIABLES.items()):
        ax = axs[0, j]
        ax.set_title(title, pad=4)
        for i, group in enumerate(REGIONS):
            annual(ax, series[series.group == group], m, WINDOW_COLORS[i], REGION_MARKERS[i])
        ax.set_xlabel("Winter year", labelpad=2)
        ax = axs[1, j]
        data = trends[trends.metric == m].set_index("group").loc[REGIONS]
        for i, r in enumerate(data.itertuples()):
            forest_mark(ax, r.slope, r.ci_low, r.ci_high, i, color=WINDOW_COLORS[i], marker=REGION_MARKERS[i])
        forest_scale(ax, WINDOW_SHORT, show_labels=j == 0, xlabel=slope)
    ax = axs[2, 0]
    for i, group in enumerate(REGIONS):
        annual(ax, series[series.group == group], "d_sca_swe", WINDOW_COLORS[i], REGION_MARKERS[i])
    ax.axhline(0, color=ZERO, lw=.65, zorder=0)
    ax.set_title("SCA–SWE contrast", pad=4)
    ax.set_ylabel("Standardized difference")
    ax.set_xlabel("Winter year", labelpad=2)
    ax = axs[2, 1]
    metric_labels = ["Mean SWE", "Volume A", "Volume B"]
    for i, group in enumerate(REGIONS):
        sub = boot[(boot.group == group) & (boot.block == 5)].set_index("swe_metric").loc[
            ["swe", "volume_a", "volume_b"]
        ]
        for row, r in enumerate(sub.itertuples()):
            forest_mark(ax, r.slope, r.ci_low, r.ci_high, row + (i - 1) * .21,
                        color=WINDOW_COLORS[i], marker=REGION_MARKERS[i])
    forest_scale(ax, metric_labels, show_labels=True, xlabel="Slope contrast (SD per decade)", xbins=3)
    ax = axs[2, 2]
    sub = dc[dc.swe_metric == "swe"].set_index("group").loc[REGIONS]
    for i, r in enumerate(sub.itertuples()):
        ax.hlines(i, r.detrended_r, r.raw_r, color=WINDOW_COLORS[i], lw=.75)
        for value, fill in [(r.raw_r, True), (r.detrended_r, False)]:
            ax.plot(value, i, linestyle="none", marker=REGION_MARKERS[i], ms=4,
                    color=WINDOW_COLORS[i], markerfacecolor=WINDOW_COLORS[i] if fill else "white",
                    markeredgewidth=.7)
    forest_scale(ax, WINDOW_SHORT, show_labels=False, xlabel="SCA–SWE Pearson r")
    ax.set_xlim(-.10, .61)
    fig.legend([Line2D([], [], color=c, marker=m, lw=.8, ms=4) for c, m in zip(WINDOW_COLORS, REGION_MARKERS)],
               WINDOW_SHORT, loc="upper center", bbox_to_anchor=(.54, .984), ncol=3, handlelength=1.8)
    fig.text(.54, .909, "Approximate geographic windows", ha="center", fontsize=8)
    save(fig, "fig05_approximate_geographic_windows", axes=axs.flat, require_panel_labels=True)


def latitude_fig(trends, temp):
    fig, axs = axes_grid(2, 104, left=.14, top=.89, bottom=.12, hspace=.65, wspace=.38)
    for j, (m, (_, slope, color)) in enumerate(VARIABLES.items()):
        data = trends[trends.metric == m].set_index("group").loc[BANDS]
        for i, r in enumerate(data.itertuples()):
            forest_mark(axs[0, j], r.slope, r.ci_low, r.ci_high, i, color=color)
        axs[0, j].set_title(m.upper() if m != "t2m" else "T2m", pad=4)
        forest_scale(axs[0, j], LAT_LABELS, show_labels=j == 0, xlabel=slope)
    ax = axs[1, 0]
    grid = temp.pivot(index="group", columns="metric", values="partial_r_year").loc[BANDS, ["scf", "swe"]].to_numpy()
    ax.imshow(grid, cmap=BALANCED, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
              extent=[-.5, 1.5, 3.5, -.5])
    for i, j in np.ndindex(grid.shape):
        ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(grid[i, j]) > .52 else INK)
    ax.set_yticks(range(len(LAT_LABELS)), LAT_LABELS)
    ax.set_xticks([0, 1], ["SCF", "SWE"])
    ax.set_title("Partial correlation (r)", pad=4)
    ax.tick_params(length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)
    for j, m in enumerate(["scf", "swe"], start=1):
        ax = axs[1, j]
        subset = temp[temp.metric == m].set_index("group").loc[BANDS]
        for i, r in enumerate(subset.itertuples()):
            forest_mark(ax, r.slope, r.ci_low, r.ci_high, i, color=VARIABLES[m][2], fill=r.q_hac <= .05)
        ax.set_title("Temperature slope", pad=4)
        units = "pp" if m == "scf" else "mm"
        forest_scale(ax, LAT_LABELS, show_labels=False, xlabel=f"{m.upper()} ({units} per °C)")
    save(fig, "fig06_latitude_response", axes=axs.flat, require_panel_labels=True)


def temperature_fig(anomaly, response, lag):
    fig, axs = axes_grid(2, 113, left=.11, top=.89, bottom=.12, wspace=.57, hspace=.56)
    groups = ["NH", *REGIONS, *BANDS]
    labels = ["NH", *WINDOW_SHORT, *LAT_LABELS]
    for i, m in enumerate(["scf", "swe"]):
        color = VARIABLES[m][2]
        ax = axs[i, 0]
        subset = anomaly[(anomaly.group == "NH") & (anomaly.metric == m)]
        x, y = subset.temp_anom.to_numpy(), subset.snow_anom.to_numpy()
        fit = stats.linregress(x, y)
        grid = np.linspace(x.min(), x.max(), 70)
        ax.scatter(x, y, s=9, alpha=.78, facecolor=color, edgecolor="none")
        ax.plot(grid, fit.intercept + fit.slope * grid, color=color, lw=1.1)
        ax.axhline(0, color=ZERO, lw=.6, zorder=0)
        ax.set_title("NH winter anomalies", pad=4)
        ax.set_xlabel("T2m anomaly (°C)", labelpad=3)
        ax.set_ylabel(f"{m.upper()} anomaly ({'pp' if m == 'scf' else 'mm'})", labelpad=3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax = axs[i, 1]
        sub = response[response.metric == m].set_index("group").loc[groups]
        for j, r in enumerate(sub.itertuples()):
            forest_mark(ax, r.partial_r_year, r.partial_ci_low_block5, r.partial_ci_high_block5,
                        j, color=color, fill=r.q_hac <= .05)
        forest_scale(ax, labels, show_labels=True, xlabel="Year-adjusted Pearson r")
        ax.axhline(.5, color="#d8dfe5", lw=.5, zorder=0)
        ax.axhline(3.5, color="#d8dfe5", lw=.5, zorder=0)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_title(f"{m.upper()}–T2m association", pad=4)
        ax = axs[i, 2]
        sub = lag[(lag.metric == m) & (lag.group == "NH")]
        ax.axhline(0, color=ZERO, lw=.65, ls=(0, (2, 2)), zorder=0)
        ax.plot(sub.lag, sub.r, color=color, lw=.8)
        for r in sub.itertuples():
            ax.vlines(r.lag, r.ci_low_block5, r.ci_high_block5, color=color, lw=.95)
            ax.plot(r.lag, r.r, color=color, marker="o", ms=4, markeredgewidth=.7,
                    markerfacecolor=color if r.q_hac_all_lags <= .05 else "white")
        ax.set_ylim(-1, 1)
        ax.set_xlim(-2.35, 2.35)
        ax.set_xticks(range(-2, 3))
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.set_title("NH cross-winter lags", pad=4)
        ax.set_xlabel("Lag (winters)", labelpad=3)
        ax.set_ylabel("Pearson r", labelpad=2)
    save(fig, "fig09_temperature_lags", axes=axs.flat, require_panel_labels=True)


def main():
    setup()
    annual_values = read("regional_latitude_metrics")
    trends = read("regional_latitude_trends")
    dc = read("area_storage_decoupling")
    boot = read("area_storage_decoupling_bootstrap")
    response = read("temperature_response")
    window_fig(annual_values, trends, dc, boot)
    latitude_fig(trends[trends.group_type == "latitude"], response[response.group_type == "latitude"])
    temperature_fig(read("temperature_anomalies"), response, read("temperature_lags"))


if __name__ == "__main__":
    main()
