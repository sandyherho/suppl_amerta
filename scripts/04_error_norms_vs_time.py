#!/usr/bin/env python
"""Time evolution of integral error norms.

Inputs : ../data/case_*.nc
Outputs: ../figs/04_error_norms_vs_time.{pdf,png,eps}
         ../stats/04_error_norms_vs_time.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "04_error_norms_vs_time"

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
            "t":        np.asarray(nc.variables["time"][:]),
            "l1_h":     np.asarray(nc.variables["l1_h"][:]),
            "l2_h":     np.asarray(nc.variables["l2_h"][:]),
            "l1_q":     np.asarray(nc.variables["l1_q"][:]),
            "l1_u_wet": np.asarray(nc.variables["l1_u_wet"][:]),
            "L":        float(nc.L),
        }


def style_axes(ax):
    ax.locator_params(axis="x", nbins=5)
    ax.tick_params(direction="out", length=3, which="major")
    ax.tick_params(direction="out", length=1.6, which="minor")
    ax.margins(x=0.02)
    ax.grid(alpha=0.3, which="both")


def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.97, top=0.97, bottom=0.13,
        hspace=0.30, wspace=0.28,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]
    tags = ["(a)", "(b)", "(c)", "(d)"]
    keys = ["l1_h", "l2_h", "l1_q", "l1_u_wet"]
    ylabels = [
        r"$L^1(h)$  [m]",
        r"$L^2(h)$  [m]",
        r"$L^1(q)$  [m$^2$ s$^{-1}$]",
        r"$L^1(u_{\mathrm{wet}})$  [m s$^{-1}$]",
    ]

    for ax, tag, key, ylab in zip(axes, tags, keys, ylabels):
        for c in CASES:
            d = data[c["key"]]
            t = d["t"]
            y = d[key]
            sel = (t > 0) & (y > 0)
            ax.plot(t[sel], y[sel], color=c["color"], lw=1.0)
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.set_title(tag, loc="left", fontweight="bold",
                     fontsize=11, pad=4)
        style_axes(ax)

    proxies = [Line2D([0], [0], color=c["color"], lw=2.0, label=c["label"])
               for c in CASES]
    fig.legend(handles=proxies, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.015), frameon=False,
               fontsize=9.5, handlelength=2.4, columnspacing=2.5)

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
        "  AMERTA  --  Time evolution of integral error norms",
        sep,
        "",
        "  Stored norms (computed by the solver against the exact",
        "  Riemann solution at every timestep):",
        "      L1(h)     = sum_x |h_num - h_an| dx          [m]",
        "      L2(h)     = sqrt(sum_x (h_num - h_an)^2 dx)  [m]",
        "      L1(q)     = sum_x |q_num - q_an| dx          [m^2/s]",
        "      L1(u_wet) = sum_x |u_num - u_an| dx          [m/s]",
        "                  restricted to cells with h > H_DRY",
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        sel = d["t"] > 0
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Snapshots stored nt", int(d["t"].size)),
            "",
            "    -- final-time values --",
            fmt("L1(h)(t_f)",     float(d["l1_h"][-1]),     "m"),
            fmt("L2(h)(t_f)",     float(d["l2_h"][-1]),     "m"),
            fmt("L1(q)(t_f)",     float(d["l1_q"][-1]),     "m^2/s"),
            fmt("L1(u_wet)(t_f)", float(d["l1_u_wet"][-1]), "m/s"),
            "",
            "    -- envelope maxima (over t > 0) --",
            fmt("max L1(h)",     float(d["l1_h"][sel].max()),     "m"),
            fmt("max L2(h)",     float(d["l2_h"][sel].max()),     "m"),
            fmt("max L1(q)",     float(d["l1_q"][sel].max()),     "m^2/s"),
            fmt("max L1(u_wet)", float(d["l1_u_wet"][sel].max()), "m/s"),
            "",
            "    -- time mean (over t > 0) --",
            fmt("<L1(h)>_t",     float(d["l1_h"][sel].mean()),     "m"),
            fmt("<L2(h)>_t",     float(d["l2_h"][sel].mean()),     "m"),
            fmt("<L1(q)>_t",     float(d["l1_q"][sel].mean()),     "m^2/s"),
            fmt("<L1(u_wet)>_t", float(d["l1_u_wet"][sel].mean()), "m/s"),
            "",
        ]
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
