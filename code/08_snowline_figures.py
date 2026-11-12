"""Publication panels for the SCF threshold boundary (Python/Matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contourpy import contour_generator
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from figure_style import FIG_WIDTH, NEUTRAL, ROOT, SCF, map_axes, map_field, panel, save, setup

CSV = ROOT / "output/csv"
AGENT = ROOT / "output/agent_snowline"
EARLY = "#262c39"
LATE = "#0e765f"
MAP_DELTA_LIMIT = 20
MAP_CMAP = "RdBu_r"
REGIONS = ("NH", "Europe", "Asia", "North America")
TITLES = ("Northern Hemisphere", "Europe window", "Asia window", "North America window")
THRESHOLDS = (.25, .50, .75)
PERIOD_LEGEND = [
    Line2D([], [], color=EARLY, ls="--", lw=1.15, label="1980–1989"),
    Line2D([], [], color=LATE, ls="-", lw=1.15, label="2013–2022"),
]


def periodic_field(lon, values):
    """Extend the longitude centres across the dateline and clip at map limits."""
    return np.r_[lon[-1] - 360, lon, lon[0] + 360], np.c_[values[:, -1], values, values[:, 0]]


def threshold_map(ax, period_fields, theta=.5, *, contours=None, lon_labels=True, lat_labels=True):
    """Draw the masked composite difference and raw early/late SCF contours."""
    lat, lon = period_fields["latitude"], period_fields["longitude"]
    lon2, delta = periodic_field(lon, period_fields["difference"])
    plot = map_field(ax, lon2, lat, 100 * delta, cmap=MAP_CMAP,
                     vmin=-MAP_DELTA_LIMIT, vmax=MAP_DELTA_LIMIT)
    for slot, color, style in [("early", EARLY, "dashed"), ("late", LATE, "solid")]:
        _, values = periodic_field(lon, period_fields[slot])
        masked = np.ma.masked_invalid(values)
        ax.contour(lon2, lat, masked, levels=[theta], colors=[color], linestyles=[style],
                   linewidths=[.72], corner_mask=False, algorithm="serial", zorder=3)
        if contours is not None:
            lines = contour_generator(x=lon2, y=lat, z=masked, corner_mask=False,
                                      line_type="Separate").lines(theta)
            for segment_id, segment in enumerate(lines):
                contours.extend({"threshold": theta, "period": slot, "segment_id": segment_id,
                                 "vertex_id": i, "longitude": xy[0], "latitude": xy[1]}
                                for i, xy in enumerate(segment))
    map_axes(ax, lon_labels=lon_labels, lat_labels=lat_labels)
    return plot


def map_colorbar(fig, ax, mappable, label, ticks, extend="neither"):
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + .05 * p.width, p.y0 - .098, .90 * p.width, .018])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, extend=extend)
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(length=2, pad=2)
    cb.outline.set_linewidth(.45)
    return cb


def draw_periods(period_fields):
    fig, grid = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.75))
    fig.subplots_adjust(left=.10, right=.97, bottom=.23, top=.94, wspace=.25, hspace=.85)
    axes = grid.ravel()
    lat, lon = period_fields["latitude"], period_fields["longitude"]
    for ax, slot, title in zip(axes, ("early", "late"), ("1980–1989", "2013–2022")):
        lon2, values = periodic_field(lon, period_fields[slot])
        im = map_field(ax, lon2, lat, 100 * values, cmap="Blues", vmin=0, vmax=100)
        map_axes(ax)
        ax.set_title(title, pad=5, x=.055)
    # The two climatology maps have identical scales. Place their shared scale between them.
    a, b = axes[:2]
    p, q = a.get_position(), b.get_position()
    cax = fig.add_axes([.5 * (p.x0 + q.x1) - .20, p.y0 - .115, .40, .017])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.set_label("SCF (%)", labelpad=2)
    cb.ax.tick_params(length=2, pad=2)
    cb.outline.set_linewidth(.45)
    contours = []
    ax = axes[2]
    im = threshold_map(ax, period_fields, contours=contours)
    ax.set_title("SCF 0.50 boundary", pad=5, x=.055)
    ax.legend(handles=PERIOD_LEGEND, loc="lower left", ncol=2, handlelength=2.0,
              columnspacing=1.1, borderaxespad=.3, borderpad=.1)
    map_colorbar(fig, ax, im, "Late minus early SCF (pp)", [-20, -10, 0, 10, 20], "both")
    ax = axes[3]
    ax.set_title("Paired longitude shifts", pad=5, x=.055)
    rows = pd.read_csv(CSV / "snowline_paired_shifts.csv").query(
        "rule == 'main' and longitude_block_deg == 10"
    )
    markers = [("#8a919c", "s"), (SCF, "o"), ("#aa6244", "D")]
    for theta, (color, marker), offset in zip(THRESHOLDS, markers, (-.22, 0, .22)):
        data = rows.loc[rows.threshold.eq(theta)].set_index("region").loc[list(REGIONS)]
        center = data.paired_median_shift_deg.to_numpy()
        ci = np.vstack([center - data.spatial_bootstrap_ci95_low,
                        data.spatial_bootstrap_ci95_high - center])
        ax.errorbar(center, np.arange(len(REGIONS)) + offset, xerr=ci, fmt=marker,
                    color=color, ms=3.1, lw=.65, capsize=1.3, label=f"{theta:.2f}")
    ax.axvline(0, color="#b4b9bf", lw=.6, zorder=0)
    ax.set_ylim(3.65, -.65)
    ax.set_xlim(-.08, 2.0)
    ax.set_yticks(np.arange(4), ["NH", "Europe", "Asia", "N. America"])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("Northward shift (° latitude)", labelpad=3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.29), ncol=3, title="SCF threshold",
              title_fontsize=6.5, handlelength=1.25, columnspacing=.9, borderaxespad=0)
    for label, ax in zip("abcd", axes):
        panel(ax, label, x=-.092, y=1.035)
    save(fig, "fig07_threshold_snowline", axes=axes, require_panel_labels=True)
    return contours


def draw_threshold_sensitivity(period_fields):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.20))
    fig.subplots_adjust(left=.095, right=.965, bottom=.23, top=.925, hspace=.40)
    contours = []
    for ax, theta, letter in zip(axes, (.25, .75), "ab"):
        im = threshold_map(ax, period_fields, theta, contours=contours)
        ax.set_title(f"SCF {theta:.2f} boundary", pad=5, x=.03)
        panel(ax, letter, x=-.051, y=1.035)
    axes[0].legend(handles=PERIOD_LEGEND, loc="lower left", ncol=2, handlelength=2.1,
                   columnspacing=1.1, borderaxespad=.35, borderpad=.1)
    map_colorbar(fig, axes[-1], im, "Late minus early SCF (pp)", [-20, -10, 0, 10, 20], "both")
    save(fig, "fig07b_snowline_threshold_sensitivity", axes=axes, require_panel_labels=True)
    return contours


def draw_annual():
    criteria = "threshold == 0.50 and rule == 'main' and support == 'available'"
    summary = pd.read_csv(CSV / "snowline_annual.csv").query(criteria)
    fits = pd.read_csv(CSV / "snowline_annual_fits.csv").query(criteria)
    fig, grid = plt.subplots(4, 2, figsize=(FIG_WIDTH, 6.6))
    fig.subplots_adjust(left=.11, right=.975, bottom=.13, top=.945, wspace=.28, hspace=.47)
    for i, (region, title) in enumerate(zip(REGIONS, TITLES)):
        left, right = grid[i]
        y = summary.loc[summary.region.eq(region)].sort_values("winter_year")
        fit = fits.loc[fits.region.eq(region)].sort_values("winter_year")
        years = y.winter_year.to_numpy()
        left.fill_between(years, fit.fitted_ci95_low.to_numpy(), fit.fitted_ci95_high.to_numpy(),
                          color=SCF, alpha=.17, linewidth=0, zorder=1)
        left.plot(years, y.median_degN, color=NEUTRAL, marker="o", ms=2.1, lw=.55, zorder=2)
        left.plot(years, fit.fitted_degN, color=SCF, lw=1.15, zorder=3)
        right.plot(years, y.n_longitudes, color=NEUTRAL, marker="o", ms=2.1, lw=.55)
        left.set_title(title, pad=4, x=.06)
        if i == 0:
            right.set_title("Available boundary longitudes", pad=4, x=.06)
        left.set_ylabel("Latitude (°N)")
        right.set_ylabel("Longitudes")
        left.yaxis.set_major_locator(MaxNLocator(4))
        right.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        for ax in (left, right):
            ax.set_xlim(1978.8, 2023.2)
            ax.set_xticks([1980, 1990, 2000, 2010, 2020])
            ax.tick_params(axis="x", labelbottom=i == 3)
            if i == 3:
                ax.set_xlabel("DJF winter")
    for label, ax in zip("abcdefgh", grid.ravel()):
        panel(ax, label, x=-.096, y=1.035)
    save(fig, "fig08_snowline_series", axes=grid.ravel(), require_panel_labels=True)


def main():
    setup()
    with np.load(AGENT / "snowline_period_fields.npz") as period_fields:
        contours = draw_periods(period_fields)
        contours.extend(draw_threshold_sensitivity(period_fields))
    pd.DataFrame(contours).to_csv(CSV / "snowline_contour_vertices.csv", index=False)
    draw_annual()


if __name__ == "__main__":
    main()
