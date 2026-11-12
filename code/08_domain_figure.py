"""Map the supplied analysis footprint and fixed geographic windows."""
import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from figure_style import FIG_WIDTH, ROOT, WINDOW_COLORS, map_axes, map_field, save, setup


def main():
    setup()
    out = ROOT / "output"
    config = json.loads((out / "analysis_config.yaml").read_text(encoding="utf-8"))
    with xr.open_dataset(out / "processed/djf_winter_fields.nc", engine="h5netcdf") as ds:
        domain = np.where(ds.analysis_mask.values == 1, 1, np.nan)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.30))
        fig.subplots_adjust(left=.09, right=.96, top=.78, bottom=.22)
        map_field(ax, ds.longitude.values, ds.latitude.values, domain, cmap=ListedColormap(["#cdd8e2"]))
        map_axes(ax)
        handles = [Patch(color="#cdd8e2", label="Analysis mask")]
        for (label, (bottom, top, left, right)), color, ls in zip(
            config["windows"].items(), WINDOW_COLORS, ["-", "--", "-."]
        ):
            rect = Rectangle((left, bottom), right-left, top-bottom, fill=False,
                             edgecolor=color, linewidth=1.0, linestyle=ls)
            ax.add_patch(rect)
            handles.append(Patch(facecolor="none", edgecolor=color, linewidth=1.0, linestyle=ls,
                                 label=label))
        ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.38))
        fig.text(.52, .065, "Approximate geographic windows", ha="center")
        save(fig, "fig01_domain")


if __name__ == "__main__":
    main()
