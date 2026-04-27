#!/usr/bin/env python
"""Phase-space (h, u) trajectories at the final time.

Each panel scatters the simulated cells of one Riemann problem in the
(h, u) state space and overlays the corresponding analytical
1-rarefaction / 1-shock and 2-rarefaction / 2-shock curves through
the left and right initial states.  Cross-case Riemann-invariant
statistics R+, R- are reported in the stats file.

Inputs : ../data/case_*.nc
Outputs: ../figs/06_phase_space_riemann_invariants.{pdf,png,eps}
         ../stats/06_phase_space_riemann_invariants.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset
from scipy.optimize import brentq


# --------------------------------------------------------------- configuration
DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "06_phase_space_riemann_invariants"

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

H_MIN_PHASE = 1.0e-3   # exclude near-floor cells from the (h, u) scatter
N_CURVE = 400          # samples per analytical phase-space branch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


# ----------------------------------------------------------------- IO helpers
def load_case(path):
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as nc:
        attrs = {k: float(getattr(nc, k)) for k in
                 ("g", "h_left", "h_right", "u_left", "u_right", "L")}
        return {
            "x":   np.asarray(nc.variables["x"][:]),
            "t":   np.asarray(nc.variables["time"][:]),
            "h":   np.asarray(nc.variables["h"][:]),
            "u":   np.asarray(nc.variables["u"][:]),
            **attrs,
        }


# --------------------------------------------------------- analytical curves
def rarefaction_curve_1(hL, uL, h_grid, g):
    cL = np.sqrt(g * hL)
    return uL + 2.0 * (cL - np.sqrt(g * h_grid))


def rarefaction_curve_2(hR, uR, h_grid, g):
    cR = np.sqrt(g * hR)
    return uR - 2.0 * (cR - np.sqrt(g * h_grid))


def shock_curve_1(hL, uL, h_grid, g):
    return uL - (h_grid - hL) * np.sqrt(
        g * (h_grid + hL) / (2.0 * h_grid * hL))


def shock_curve_2(hR, uR, h_grid, g):
    return uR + (h_grid - hR) * np.sqrt(
        g * (h_grid + hR) / (2.0 * h_grid * hR))


# --------------------------------------------------------- star-state solvers
def stoker_star(hL, hR, uL, uR, g):
    cL = np.sqrt(g * hL)

    def fn(h):
        c = np.sqrt(g * h)
        return (uL + 2.0 * (cL - c)) - (uR + (h - hR) * np.sqrt(
            g * (h + hR) / (2.0 * h * hR)))

    h_star = brentq(fn, hR * 1.0001, hL * 0.9999, xtol=1e-12, maxiter=200)
    u_star = uL + 2.0 * (cL - np.sqrt(g * h_star))
    return h_star, u_star


def double_shock_star(h0, U, g):
    def fn(h):
        return U - (h - h0) * np.sqrt(g * (h + h0) / (2.0 * h * h0))
    return brentq(fn, h0 * 1.0001, h0 * 500.0, xtol=1e-12, maxiter=200)


def double_rarefaction_star(h0, U, g):
    c0 = np.sqrt(g * h0)
    c_star = c0 - 0.5 * U
    return max(c_star ** 2 / g, 0.0)


def analytical_curves(case_key, d):
    g = d["g"]
    hL, hR = d["h_left"], d["h_right"]
    uL, uR = d["u_left"], d["u_right"]
    curves = []

    if case_key == "stoker":
        h_star, _ = stoker_star(hL, hR, uL, uR, g)
        h_rar = np.linspace(h_star, hL, N_CURVE)
        h_sho = np.linspace(hR, h_star, N_CURVE)
        curves.append(("1-rarefaction", h_rar,
                       rarefaction_curve_1(hL, uL, h_rar, g)))
        curves.append(("2-shock", h_sho,
                       shock_curve_2(hR, uR, h_sho, g)))

    elif case_key == "ritter":
        h_rar = np.linspace(1.0e-4, hL, N_CURVE)
        curves.append(("1-rarefaction", h_rar,
                       rarefaction_curve_1(hL, uL, h_rar, g)))

    elif case_key == "double_rarefaction":
        h_star = double_rarefaction_star(hL, abs(uL), g)
        h_lo = max(h_star, 1.0e-4)
        h1 = np.linspace(h_lo, hL, N_CURVE)
        h2 = np.linspace(h_lo, hR, N_CURVE)
        curves.append(("1-rarefaction", h1,
                       rarefaction_curve_1(hL, uL, h1, g)))
        curves.append(("2-rarefaction", h2,
                       rarefaction_curve_2(hR, uR, h2, g)))

    elif case_key == "double_shock":
        h_star = double_shock_star(hL, uL, g)
        h1 = np.linspace(hL, h_star, N_CURVE)
        h2 = np.linspace(hR, h_star, N_CURVE)
        curves.append(("1-shock", h1, shock_curve_1(hL, uL, h1, g)))
        curves.append(("2-shock", h2, shock_curve_2(hR, uR, h2, g)))

    return curves


def nearest_curve_distance(h_pts, u_pts, curves):
    if not curves:
        return np.full_like(h_pts, np.nan)
    all_h = np.concatenate([c[1] for c in curves])
    all_u = np.concatenate([c[2] for c in curves])
    d = np.full_like(h_pts, np.inf, dtype=float)
    for i, (hp, up) in enumerate(zip(h_pts, u_pts)):
        d[i] = np.min((all_h - hp) ** 2 + (all_u - up) ** 2)
    return np.sqrt(d)


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.97, top=0.94, bottom=0.13,
        hspace=0.42, wspace=0.30,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        d = data[c["key"]]
        h_pts = d["h"][-1]
        u_pts = d["u"][-1]
        m = h_pts > H_MIN_PHASE
        ax.plot(h_pts[m], u_pts[m], "o", ms=2.6, mfc="none",
                mec=c["color"], mew=0.7)
        for _, h_arr, u_arr in analytical_curves(c["key"], d):
            ax.plot(h_arr, u_arr, color="k", lw=1.0, alpha=0.85)
        ax.axhline(0.0, color="gray", lw=0.4, ls=":")
        ax.set_xlabel(r"$h$  [m]")
        ax.set_ylabel(r"$u$  [m s$^{-1}$]")
        ax.set_title(tag, fontsize=11, fontweight="bold", pad=6)
        ax.grid(alpha=0.3)

    proxies = [
        Line2D([0], [0], marker="o", color=CASES[0]["color"], lw=0,
               mfc="none", mec=CASES[0]["color"], mew=0.9, ms=6,
               label=CASES[0]["label"]),
        Line2D([0], [0], marker="o", color=CASES[1]["color"], lw=0,
               mfc="none", mec=CASES[1]["color"], mew=0.9, ms=6,
               label=CASES[1]["label"]),
        Line2D([0], [0], marker="o", color=CASES[2]["color"], lw=0,
               mfc="none", mec=CASES[2]["color"], mew=0.9, ms=6,
               label=CASES[2]["label"]),
        Line2D([0], [0], marker="o", color=CASES[3]["color"], lw=0,
               mfc="none", mec=CASES[3]["color"], mew=0.9, ms=6,
               label=CASES[3]["label"]),
        Line2D([0], [0], color="k", lw=1.4, label="Analytical curves"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------- stats
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
        "  AMERTA v0.0.3  --  Phase-space and Riemann invariants",
        sep,
        "",
        "  At t_final each cell is plotted in (h, u) state space and",
        "  compared to the analytical wave curves emanating from the",
        "  left and right initial states.  Riemann invariants",
        "      R+ = u + 2 sqrt(g h)   (right-going characteristics)",
        "      R- = u - 2 sqrt(g h)   (left-going characteristics)",
        "  are conserved across the corresponding rarefaction fans.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Phase-space cell filter h >", H_MIN_PHASE, "m"),
        fmt("Samples per analytical branch", N_CURVE),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        h_pts = d["h"][-1]
        u_pts = d["u"][-1]
        m = h_pts > H_MIN_PHASE
        cL = np.sqrt(d["g"] * h_pts[m])
        Rp = u_pts[m] + 2.0 * cL
        Rm = u_pts[m] - 2.0 * cL
        curves = analytical_curves(c["key"], d)
        dist = nearest_curve_distance(h_pts[m], u_pts[m], curves)

        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            fmt(f"Active cells (h > {H_MIN_PHASE:.0e} m)",
                int(m.sum()), f"/ {len(h_pts)}"),
            "",
            "    -- Riemann invariant R+ = u + 2 sqrt(g h) --",
            fmt("min", float(Rp.min()), "m s^-1"),
            fmt("max", float(Rp.max()), "m s^-1"),
            fmt("mean", float(Rp.mean()), "m s^-1"),
            fmt("std", float(Rp.std()), "m s^-1"),
            "",
            "    -- Riemann invariant R- = u - 2 sqrt(g h) --",
            fmt("min", float(Rm.min()), "m s^-1"),
            fmt("max", float(Rm.max()), "m s^-1"),
            fmt("mean", float(Rm.mean()), "m s^-1"),
            fmt("std", float(Rm.std()), "m s^-1"),
        ]
        if curves:
            lines += [
                "",
                "    -- distance to nearest analytical (h, u) curve --",
                fmt("median", float(np.median(dist)), ""),
                fmt("mean", float(np.mean(dist)), ""),
                fmt("p95", float(np.percentile(dist, 95)), ""),
                fmt("max", float(np.max(dist)), ""),
            ]
        lines.append("")
    outpath.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
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
