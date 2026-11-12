"""Integrate the verified trend and boundary fields and show the evidence boundary."""
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from figure_style import FIG_WIDTH, INK, NEUTRAL, ROOT, SCF, SWE, T2M, map_axes, panel, save, setup

trend = import_module("08_trend_figures")
snowline = import_module("08_snowline_figures")


def colorbar(fig, ax, mappable, label, ticks, extend):
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + .065 * box.width, box.y0 - .082, .87 * box.width, .017])
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, extend=extend)
    bar.set_label(label, labelpad=1)
    bar.ax.tick_params(labelsize=6.5, pad=1, length=2)
    bar.outline.set_linewidth(.5)


def synthesis():
    out = ROOT / "output"
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.55))
    fig.subplots_adjust(left=.096, right=.962, bottom=.26, top=.925, wspace=.255, hspace=.93)
    with xr.open_dataset(out / "processed/grid_trends.nc", engine="h5netcdf") as grid:
        for ax, letter, key, title in zip(axs.flat, "abc", ("scf", "swe", "t2m"),
                                           ("Snow cover trend", "Snow water equivalent trend",
                                            "2 m air temperature trend")):
            mappable, label, ticks, extend = trend.trend_map(ax, grid, key)
            map_axes(ax)
            colorbar(fig, ax, mappable, label, ticks, extend)
            ax.set_title(title, pad=5)
            panel(ax, letter, -.105, 1.04)
    ax = axs[1, 1]
    with np.load(out / "agent_snowline/snowline_period_fields.npz") as data:
        mappable = snowline.threshold_map(ax, data)
    colorbar(fig, ax, mappable, "SCF difference (pp)", [-20, -10, 0, 10, 20], "both")
    ax.set_title("SCF 0.50 boundary", pad=5)
    panel(ax, "d", -.105, 1.04)
    fig.legend(handles=snowline.PERIOD_LEGEND, loc="lower right", bbox_to_anchor=(.952, .027),
               ncol=2, handlelength=2.3)
    save(fig, "fig10_synthesis", axes=list(axs.flat))


def interpretation():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 1.75))
    fig.subplots_adjust(left=.04, right=.96, bottom=.04, top=.96)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    for x, label, color in ((.16, "Snow cover fraction (SCF)", SCF),
                            (.50, "2 m air temperature (T2m)", T2M),
                            (.84, "Snow water equivalent (SWE)", SWE)):
        ax.text(x, .69, label, ha="center", va="center", color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=.45", ec=color, fc="white", lw=.8))
    for left, right, label in [(.16, .5, "Negative residual\nassociation"),
                               (.5, .84, "Positive residual\nassociation")]:
        ax.plot([left, left, right, right], [.79, .83, .83, .79], color=INK, linewidth=.75)
        ax.text((left+right)/2, .865, label, fontsize=6.5, ha="center", va="bottom")
    ax.plot([.16, .16, .84, .84], [.55, .34, .34, .55], ls=(0,(3,2)), color=NEUTRAL, linewidth=.75)
    ax.plot([.5, .5], [.55, .34], ls=(0,(3,2)), color=NEUTRAL, linewidth=.75)
    ax.text(.5, .22, "Hypotheses involving unobserved processes", ha="center", weight="bold")
    ax.text(.5, .09, "Snow preservation, melt and precipitation phase", ha="center", color=NEUTRAL)
    save(fig, "fig11_interpretation")


if __name__ == "__main__":
    setup()
    synthesis()
    interpretation()
