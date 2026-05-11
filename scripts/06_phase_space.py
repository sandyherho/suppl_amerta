#!/usr/bin/env python
"""Phase-space (h, u) at final time vs analytical wave curves.

Inputs : ../data/case_*.nc
Outputs: ../figs/06_phase_space.{pdf,png,eps}
         ../stats/06_phase_space.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset
from scipy.optimize import brentq


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "06_phase_space"

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

H_MIN_PHASE = 1.0e-3
N_CURVE     = 400

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


def rarefaction_1(hL, uL, h_grid, g):
    cL = np.sqrt(g * hL)
    return uL + 2.0 * (cL - np.sqrt(g * h_grid))


def rarefaction_2(hR, uR, h_grid, g):
    cR = np.sqrt(g * hR)
    return uR - 2.0 * (cR - np.sqrt(g * h_grid))


def shock_1(hL, uL, h_grid, g):
    return uL - (h_grid - hL) * np.sqrt(
        g * (h_grid + hL) / (2.0 * h_grid * hL))


def shock_2(hR, uR, h_grid, g):
    return uR + (h_grid - hR) * np.sqrt(
        g * (h_grid + hR) / (2.0 * h_grid * hR))


def stoker_star(hL, hR, uL, uR, g):
    cL = np.sqrt(g * hL)

    def fn(h):
        c = np.sqrt(g * h)
        return ((uL + 2.0 * (cL - c))
                - (uR + (h - hR) * np.sqrt(g * (h + hR) / (2.0 * h * hR))))
    h_star = brentq(fn, hR * 1.0001, hL * 0.9999, xtol=1e-12, maxiter=200)
    u_star = uL + 2.0 * (cL - np.sqrt(g * h_star))
    return h_star, u_star


def double_rarefaction_star(h0, U, g):
    c0 = np.sqrt(g * h0)
    c_star = c0 - 0.5 * U
    return max(c_star ** 2 / g, 0.0), 0.0


def double_shock_star(h0, U, g):
    def fn(h):
        return U - (h - h0) * np.sqrt(g * (h + h0) / (2.0 * h * h0))
    h_star = brentq(fn, h0 * 1.0001, h0 * 500.0, xtol=1e-12, maxiter=200)
    return h_star, 0.0


def analytical_curves(case_key, hL, hR, uL, uR, g):
    if case_key == "stoker":
        h_star, u_star = stoker_star(hL, hR, uL, uR, g)
        h_rar = np.linspace(h_star, hL, N_CURVE)
        h_sho = np.linspace(hR, h_star, N_CURVE)
        return ([("1-rarefaction", h_rar, rarefaction_1(hL, uL, h_rar, g)),
                 ("2-shock",       h_sho, shock_2      (hR, uR, h_sho, g))],
                (h_star, u_star))
    if case_key == "ritter":
        h_rar = np.linspace(1.0e-4, hL, N_CURVE)
        return ([("1-rarefaction", h_rar,
                  rarefaction_1(hL, uL, h_rar, g))],
                (None, None))
    if case_key == "double_rarefaction":
        h_star, u_star = double_rarefaction_star(hL, abs(uL), g)
        h_lo = max(h_star, 1.0e-4)
        h1 = np.linspace(h_lo, hL, N_CURVE)
        h2 = np.linspace(h_lo, hR, N_CURVE)
        return ([("1-rarefaction", h1, rarefaction_1(hL, uL, h1, g)),
                 ("2-rarefaction", h2, rarefaction_2(hR, uR, h2, g))],
                (h_star, u_star))
    if case_key == "double_shock":
        h_star, u_star = double_shock_star(hL, uL, g)
        h1 = np.linspace(hL, h_star, N_CURVE)
        h2 = np.linspace(hR, h_star, N_CURVE)
        return ([("1-shock", h1, shock_1(hL, uL, h1, g)),
                 ("2-shock", h2, shock_2(hR, uR, h2, g))],
                (h_star, u_star))
    return [], (None, None)


def nearest_curve_distance(h_pts, u_pts, curves):
    if not curves or len(h_pts) == 0:
        return np.full_like(h_pts, np.nan, dtype=float)
    all_h = np.concatenate([c[1] for c in curves])
    all_u = np.concatenate([c[2] for c in curves])
    d = np.empty_like(h_pts, dtype=float)
    for i, (hp, up) in enumerate(zip(h_pts, u_pts)):
        d[i] = np.min((all_h - hp) ** 2 + (all_u - up) ** 2)
    return np.sqrt(d)


def load_case(path):
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as nc:
        return {
            "h":  np.asarray(nc.variables["h"][-1]),
            "u":  np.asarray(nc.variables["u"][-1]),
            "g":  float(nc.g),
            "L":  float(nc.L),
            "hL": float(nc.h_left),
            "hR": float(nc.h_right),
            "uL": float(nc.u_left),
            "uR": float(nc.u_right),
            "tf": float(nc.t_final),
        }


def style_axes(ax):
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=6)
    ax.tick_params(direction="out", length=3)
    ax.margins(x=0.05, y=0.08)
    ax.grid(alpha=0.3)


def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.97, bottom=0.13,
        hspace=0.30, wspace=0.24,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        d = data[c["key"]]
        h, u = d["h"], d["u"]
        m = h > H_MIN_PHASE
        ax.plot(h[m], u[m], "o", ms=2.6, mfc="none",
                mec=c["color"], mew=0.7, zorder=3)
        curves, (h_star, u_star) = analytical_curves(
            c["key"], d["hL"], d["hR"], d["uL"], d["uR"], d["g"])
        for _, h_arr, u_arr in curves:
            ax.plot(h_arr, u_arr, color="k", lw=1.2, alpha=0.85, zorder=4)
        ax.plot(d["hL"], d["uL"], marker="*", ms=12,
                mec="k", mfc=c["color"], mew=0.7, zorder=5)
        ax.plot(d["hR"], d["uR"], marker="*", ms=12,
                mec="k", mfc=c["color"], mew=0.7, zorder=5)
        if h_star is not None:
            ax.plot(h_star, u_star, marker="D", ms=7,
                    mec="k", mfc="white", mew=0.9, zorder=6)
        ax.axhline(0.0, color="gray", lw=0.4, ls=":")
        ax.set_xlabel(r"$h$  [m]")
        ax.set_ylabel(r"$u$  [m s$^{-1}$]")
        ax.set_title(tag, loc="left", fontweight="bold",
                     fontsize=11, pad=4)
        style_axes(ax)

    proxies = [Line2D([0], [0], marker="o", color="w", mfc="none",
                      mec=c["color"], mew=0.9, ms=6, label=c["label"])
               for c in CASES]
    proxies += [
        Line2D([0], [0], color="k", lw=1.4, label="analytical wave curves"),
        Line2D([0], [0], marker="*", color="w", mfc="white", mec="k",
               mew=0.7, ms=11, label=r"initial states"),
        Line2D([0], [0], marker="D", color="w", mfc="white", mec="k",
               mew=0.9, ms=7, label=r"star state $(h_*, u_*)$"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, 0.015), frameon=False,
               fontsize=9.0, handlelength=2.2, columnspacing=1.5)

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
        "  AMERTA  --  Phase-space (h, u) at t_f vs analytical wave curves",
        sep,
        "",
        "  Wave curves connect (h_L, u_L) and (h_R, u_R) through the",
        "  star state (h_*, u_*).  At t_f the simulated cells should",
        "  cluster on these curves.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Phase-space cell filter h >", H_MIN_PHASE, "m"),
        fmt("Samples per analytical wave curve", N_CURVE),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        h, u = d["h"], d["u"]
        m = h > H_MIN_PHASE
        curves, (h_star, u_star) = analytical_curves(
            c["key"], d["hL"], d["hR"], d["uL"], d["uR"], d["g"])
        dist = nearest_curve_distance(h[m], u[m], curves)
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Final time t_f", float(d["tf"]), "s"),
            fmt(f"Active cells (h > {H_MIN_PHASE:.0e} m)",
                int(m.sum()), f"/ {len(h)}"),
            "",
            "    -- initial / star states --",
            fmt("h_L", float(d["hL"]), "m"),
            fmt("h_R", float(d["hR"]), "m"),
            fmt("u_L", float(d["uL"]), "m/s"),
            fmt("u_R", float(d["uR"]), "m/s"),
        ]
        if h_star is not None:
            lines += [
                fmt("h_* (analytical)", float(h_star), "m"),
                fmt("u_* (analytical)", float(u_star), "m/s"),
            ]
        else:
            lines.append(
                "    star state                                 "
                "(not applicable for Ritter)")
        if dist.size and not np.all(np.isnan(dist)):
            lines += [
                "",
                "    -- distance from each cell to nearest analytical curve --",
                fmt("median", float(np.median(dist)), ""),
                fmt("mean",   float(np.mean(dist)),   ""),
                fmt("p95",    float(np.percentile(dist, 95)), ""),
                fmt("max",    float(np.max(dist)),    ""),
            ]
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
