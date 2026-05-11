#!/usr/bin/env python
"""Space-time surfaces of h(x, t).

Inputs : ../data/case_*.nc
Outputs: ../figs/01_spacetime_surfaces.{pdf,png,eps}
         ../stats/01_spacetime_surfaces.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
from netCDF4 import Dataset


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "01_spacetime_surfaces"

CASES = [
    {"file": "case_1_stoker_wet_dam_break.nc", "key": "stoker",
     "label": "Stoker"},
    {"file": "case_2_ritter_dry_dam_break.nc", "key": "ritter",
     "label": "Ritter"},
    {"file": "case_3_double_rarefaction.nc",   "key": "double_rarefaction",
     "label": "Double rarefaction"},
    {"file": "case_4_double_shock.nc",         "key": "double_shock",
     "label": "Double shock"},
]

NT_PLOT = 80
NX_PLOT = 200
ELEV    = 28
AZIM    = -60
CMAP    = "viridis"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "axes.labelpad": 2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def load_h_field(path):
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as nc:
        return {
            "x": np.asarray(nc.variables["x"][:]),
            "t": np.asarray(nc.variables["time"][:]),
            "h": np.asarray(nc.variables["h"][:]),
            "L": float(nc.L),
        }


def subsample(x, t, h, nx_plot, nt_plot):
    ix = np.linspace(0, len(x) - 1, min(nx_plot, len(x))).astype(int)
    it = np.linspace(0, len(t) - 1, min(nt_plot, len(t))).astype(int)
    return x[ix], t[it], h[np.ix_(it, ix)]


def plot_surfaces(cases, vmin, vmax, outpath_stem):
    fig = plt.figure(figsize=(9.4, 7.6))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.035],
        left=0.03, right=0.93, top=0.97, bottom=0.04,
        hspace=0.12, wspace=0.08,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        d = cases[c["key"]]
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection="3d")
        X, T = np.meshgrid(d["x_sub"], d["t_sub"])
        ax.plot_surface(
            X, T, d["h_sub"],
            cmap=CMAP, vmin=vmin, vmax=vmax,
            rstride=1, cstride=1, linewidth=0,
            antialiased=False, edgecolor="none",
        )
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$t$  [s]")
        ax.set_zlabel(r"$h$  [m]")
        ax.set_zlim(vmin, vmax)
        ax.set_xticks([0, 500, 1000, 1500, 2000])

        # t-ticks per case: 5 evenly spaced on [0, t_final]
        t_final = float(d["t_sub"][-1])
        if t_final >= 70:        # 80-second runs
            ax.set_yticks([0, 20, 40, 60, 80])
        else:                    # Ritter (40 s)
            ax.set_yticks([0, 10, 20, 30, 40])

        ax.set_zticks([2, 4, 6, 8, 10])
        ax.tick_params(pad=0)
        ax.text2D(0.0, 1.0, tag, transform=ax.transAxes,
                  fontsize=11, fontweight="bold", va="top", ha="left")

    cax = fig.add_subplot(gs[:, 2])
    sm = cm.ScalarMappable(cmap=CMAP,
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$h$  [m]")

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


def fmt(label, value, unit=""):
    if isinstance(value, (int, np.integer)):
        s = f"{int(value):>16d}"
    elif isinstance(value, float):
        if value != value:
            s = f"{'NaN':>16}"
        elif abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-3):
            s = f"{value:>16.6e}"
        else:
            s = f"{value:>16.6f}"
    else:
        s = f"{str(value):>16}"
    return f"    {label:<46}{s}  {unit}".rstrip()


def write_stats(cases, outpath, vmin, vmax):
    sep = "=" * 80
    sub = "-" * 80
    lines = [
        sep,
        "  AMERTA  --  Space-time surfaces of h(x, t)",
        sep,
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Global colormap range (min)", vmin, "m"),
        fmt("Global colormap range (max)", vmax, "m"),
        fmt("Plot subsampling along t",    NT_PLOT, ""),
        fmt("Plot subsampling along x",    NX_PLOT, ""),
        fmt("View elevation",              ELEV, "deg"),
        fmt("View azimuth",                AZIM, "deg"),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = cases[c["key"]]
        x, t, h = d["x"], d["t"], d["h"]
        idx_max = np.unravel_index(int(np.argmax(h)), h.shape)
        idx_min = np.unravel_index(int(np.argmin(h)), h.shape)
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Snapshots stored nt", int(h.shape[0])),
            fmt("Grid points nx",      int(h.shape[1])),
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("t range (min)", float(t[0]), "s"),
            fmt("t range (max)", float(t[-1]), "s"),
            fmt("x range (min)", float(x[0]), "m"),
            fmt("x range (max)", float(x[-1]), "m"),
            fmt("h_min",                  float(h.min()),  "m"),
            fmt("argmin h x location",    float(x[idx_min[1]]), "m"),
            fmt("argmin h t location",    float(t[idx_min[0]]), "s"),
            fmt("h_max",                  float(h.max()),  "m"),
            fmt("argmax h x location",    float(x[idx_max[1]]), "m"),
            fmt("argmax h t location",    float(t[idx_max[0]]), "s"),
            fmt("h_mean over (x, t)",     float(h.mean()), "m"),
            "",
        ]
    outpath.write_text("\n".join(lines) + "\n")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    cases = {}
    for c in CASES:
        d = load_h_field(DATA_DIR / c["file"])
        x_s, t_s, h_s = subsample(d["x"], d["t"], d["h"], NX_PLOT, NT_PLOT)
        cases[c["key"]] = {**d, "x_sub": x_s, "t_sub": t_s, "h_sub": h_s}
    vmin = float(min(c["h"].min() for c in cases.values()))
    vmax = float(max(c["h"].max() for c in cases.values()))
    plot_surfaces(cases, vmin, vmax, FIGS_DIR / OUTPUT_NAME)
    write_stats(cases, STATS_DIR / f"{OUTPUT_NAME}.txt", vmin, vmax)
    print(f"wrote: {FIGS_DIR / OUTPUT_NAME}.{{pdf,png,eps}}")
    print(f"wrote: {STATS_DIR / OUTPUT_NAME}.txt")


if __name__ == "__main__":
    main()
