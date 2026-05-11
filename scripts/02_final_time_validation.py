#!/usr/bin/env python
"""Final-time numerical-vs-analytical depth profile.

For each Riemann case, overlays h_num(x, t_f) (color, solid) and
h_an(x, t_f) (black, dashed).

Inputs : ../data/case_*.nc
Outputs: ../figs/02_final_time_validation.{pdf,png,eps}
         ../stats/02_final_time_validation.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "02_final_time_validation"

CASES = [
    {"file": "case_1_stoker_wet_dam_break.nc", "key": "stoker",
     "label": "Stoker",             "color": "#1f77b4"},
    {"file": "case_2_ritter_dry_dam_break.nc", "key": "ritter",
     "label": "Ritter",             "color": "#ff7f0e"},
    {"file": "case_3_double_rarefaction.nc",   "key": "double_rarefaction",
     "label": "Double rarefaction", "color": "#2ca02c"},
    {"file": "case_4_double_shock.nc",         "key": "double_shock",
     "label": "Double shock",       "color": "#d62728"},
]

ERR_PERCENTILES = (50, 75, 90, 95, 99, 99.9)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})


def load_case(path):
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as nc:
        return {
            "x":     np.asarray(nc.variables["x"][:]),
            "h":     np.asarray(nc.variables["h"][-1]),
            "h_an":  np.asarray(nc.variables["h_analytical"][-1]),
            "h_err": np.asarray(nc.variables["h_error"][-1]),
            "l1_h":  float(nc.variables["l1_h"][-1]),
            "l2_h":  float(nc.variables["l2_h"][-1]),
            "L":     float(nc.L),
            "t_f":   float(nc.variables["time"][-1]),
        }


def style_axes(ax):
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=6)
    ax.tick_params(direction="out", length=3)
    ax.margins(x=0.02, y=0.06)
    ax.grid(alpha=0.3)


def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.97, bottom=0.13,
        hspace=0.30, wspace=0.22,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        d = data[c["key"]]
        ax.plot(d["x"], d["h_an"], color="k", lw=1.4, ls="--",
                alpha=0.9, zorder=3)
        ax.plot(d["x"], d["h"], color=c["color"], lw=1.0, zorder=4)
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$h$  [m]")
        ax.set_title(tag, loc="left", fontweight="bold",
                     fontsize=11, pad=4)
        style_axes(ax)

    proxies = [Line2D([0], [0], color=c["color"], lw=2.0, label=c["label"])
               for c in CASES]
    proxies.append(Line2D([0], [0], color="k", lw=1.4, ls="--",
                          label="analytical"))
    fig.legend(handles=proxies, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.015), frameon=False,
               fontsize=9.5, handlelength=2.4, columnspacing=2.0)

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


def write_stats(data, outpath):
    sep = "=" * 80
    sub = "-" * 80
    lines = [
        sep,
        "  AMERTA  --  Final-time numerical-vs-analytical validation",
        sep,
        "",
        "  Pointwise error e(x) = h_num(x, t_f) - h_an(x, t_f).",
        "  L1, L2 are the values stored in each NetCDF and computed",
        "  by the solver against the exact Riemann solution.",
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        err   = np.abs(d["h_err"])
        i_max = int(np.argmax(err))
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t_f"]), "s"),
            fmt("Grid points nx", int(d["x"].size)),
            "",
            "    -- final-time integral norms (from NetCDF) --",
            fmt("L1(h) at t_f", float(d["l1_h"]), "m"),
            fmt("L2(h) at t_f", float(d["l2_h"]), "m"),
            "",
            "    -- pointwise |h_err|(x) at t_f --",
            fmt("max", float(err.max()), "m"),
            fmt("mean", float(err.mean()), "m"),
            fmt("argmax x location", float(d["x"][i_max]), "m"),
        ]
        for p in ERR_PERCENTILES:
            lines.append(fmt(f"{p:5.1f}-th percentile",
                             float(np.percentile(err, p)), "m"))
        lines.append("")
    outpath.write_text("\n".join(lines) + "\n")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    data = {c["key"]: load_case(DATA_DIR / c["file"]) for c in CASES}
    plot_figure(data, FIGS_DIR / OUTPUT_NAME)
    write_stats(data, STATS_DIR / f"{OUTPUT_NAME}.txt")
    print(f"wrote: {FIGS_DIR / OUTPUT_NAME}.{{pdf,png,eps}}")
    print(f"wrote: {STATS_DIR / OUTPUT_NAME}.txt")


if __name__ == "__main__":
    main()
