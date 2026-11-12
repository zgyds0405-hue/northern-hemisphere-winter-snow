"""Shared Python publication figure style and render-time preflight."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "output" / "figures"
QA = ROOT / "logs" / "figure_qa"
DEFAULT_SKILL = (Path.home() / ".codex/plugins/cache/nature-skills/nature-skills"
                 / "0.1.0/skills/nature-figure")
SKILL = Path(os.environ.get("NATURE_FIGURE_SKILL", DEFAULT_SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

from audit_panel_alignment import require_matplotlib_panel_alignment

INK = "#233248"
NEUTRAL = "#556273"
SCF = "#235f85"
SWE = "#16786b"
T2M = "#ac5235"
WINDOW_COLORS = ["#235f85", "#ad582e", "#16786b"]
FIG_WIDTH = 160 / 25.4


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0, "axes.titlesize": 8.0, "axes.labelsize": 7.0,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .60,
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.labelcolor": INK, "axes.edgecolor": NEUTRAL, "text.color": INK,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.color": INK,
        "ytick.color": INK, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.width": .6, "ytick.major.width": .6, "legend.fontsize": 6.5,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, letter, x=-.095, y=1.04):
    ax.set_gid(letter)
    artist = ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                     va="bottom", ha="left", clip_on=False)
    artist.set_gid("panel-label")
    return artist


def _longitude(value, _pos):
    if value in (-180, 180):
        return "180°"
    return f"{abs(value):.0f}°" + ("W" if value < 0 else "E" if value > 0 else "")


def _latitude(value, _pos):
    return f"{value:.0f}°" + ("N" if value > 0 else "")


def map_axes(ax, lon_labels=True, lat_labels=True):
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 90)
    ax.xaxis.set_major_locator(FixedLocator([-180, -120, -60, 0, 60, 120, 180]))
    ax.yaxis.set_major_locator(FixedLocator([0, 30, 60, 90]))
    ax.xaxis.set_major_formatter(FuncFormatter(_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(_latitude))
    ax.tick_params(labelbottom=lon_labels, labelleft=lat_labels)
    return ax


def map_field(ax, lon, lat, values, *, cmap, **kwargs):
    field = np.ma.masked_invalid(values)
    return ax.pcolormesh(lon, lat, field, cmap=cmap, shading="nearest", rasterized=True, **kwargs)


def save(fig, stem, *, axes=None, **alignment_options):
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig, axes=axes, json_out=QA / f"{stem}.alignment.json",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       strict=True, **alignment_options)
    fig.savefig(FIGURES / f"{stem}.pdf", dpi=600)
    fig.savefig(FIGURES / f"{stem}.svg", dpi=300)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600)
    plt.close(fig)
