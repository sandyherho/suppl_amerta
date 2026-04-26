#!/usr/bin/env python
"""Phase-space (h, u) trajectories and Riemann-invariant clustering.

Each Riemann solution traces specific curves in the conservative
state space: 1- and 2-rarefaction integral curves preserve the
Riemann invariants R_+ = u + 2*sqrt(g h) (right-going characteristics)
and R_- = u - 2*sqrt(g h) (left-going characteristics) respectively;
shocks lie on Hugoniot loci.  This script overlays the simulated
state at the final time onto the analytical curves and shows the
joint distribution of Riemann invariants across cases.

Inputs : ../data/case_*.nc
Outputs: ../figs/06_phase_space_riemann_invariants.{pdf,png,eps}
         ../stats/06_phase_space_riemann_invariants.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

H_MIN_PHASE = 1.0e-3   # ignore near-floor cells when scattering (h, u)
N_CURVE = 400          # samples for the analytical phase-space curves

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
    """1-rarefaction: u = uL + 2*(c_L - c)  (R+ conserved)."""
    cL = np.sqrt(g * hL)
    return uL + 2.0 * (cL - np.sqrt(g * h_grid))


def rarefaction_curve_2(hR, uR, h_grid, g):
    """2-rarefaction: u = uR - 2*(c_R - c)  (R- conserved)."""
    cR = np.sqrt(g * hR)
    return uR - 2.0 * (cR - np.sqrt(g * h_grid))


def shock_curve_2(hR, uR, h_grid, g):
    """2-shock Hugoniot: u = uR + (h - hR) * sqrt(g*(h+hR)/(2*h*hR))."""
    return uR + (h_grid - hR) * np.sqrt(
        g * (h_grid + hR) / (2.0 * h_grid * hR))


def shock_curve_1(hL, uL, h_grid, g):
    """1-shock Hugoniot: u = uL - (h - hL) * sqrt(g*(h+hL)/(2*h*hL))."""
    return uL - (h_grid - hL) * np.sqrt(
        g * (h_grid + hL) / (2.0 * h_grid * hL))


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


# ------------------------------------------- analytical curves per Riemann ic
def analytical_curves(case_key, d):
    """Return list of (label, h_arr, u_arr) ready to plot."""
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
        if h_star <= 0:
            h_star = 1.0e-4
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
    """Min distance from each (h, u) to the union of analytical curves."""
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
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(
        2, 3, left=0.07, right=0.985, top=0.97, bottom=0.08,
        hspace=0.32, wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[:, 2])]
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # (a)-(d): per-case (h, u) scatter + analytical curves
    for ax, tag, c in zip(axes[:4], tags[:4], CASES):
        d = data[c["key"]]
        h_pts = d["h"][-1]
        u_pts = d["u"][-1]
        m = h_pts > H_MIN_PHASE
        ax.plot(h_pts[m], u_pts[m], "o", ms=2.0, mfc="none",
                mec=c["color"], mew=0.6, label="numerical")
        for name, h_arr, u_arr in analytical_curves(c["key"], d):
            ax.plot(h_arr, u_arr, "k-", lw=1.0, alpha=0.85, label=name)
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_xlabel(r"$h$ [m]")
        ax.set_ylabel(r"$u$ [m s$^{-1}$]")
        ax.text(0.03, 0.95, tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="best", fontsize=7, frameon=False)
        ax.grid(alpha=0.3)

    # (e) (R+, R-) joint scatter for all cases
    ax = axes[4]
    for c in CASES:
        d = data[c["key"]]
        h_pts = d["h"][-1]
        u_pts = d["u"][-1]
        m = h_pts > H_MIN_PHASE
        cL = np.sqrt(d["g"] * h_pts[m])
        Rp = u_pts[m] + 2.0 * cL
        Rm = u_pts[m] - 2.0 * cL
        ax.plot(Rp, Rm, "o", ms=1.8, mfc="none",
                mec=c["color"], mew=0.6, label=c["label"])
    ax.set_xlabel(r"$R_+ = u + 2\sqrt{gh}$ [m s$^{-1}$]")
    ax.set_ylabel(r"$R_- = u - 2\sqrt{gh}$ [m s$^{-1}$]")
    ax.text(0.03, 0.95, tags[4], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------- stats
def write_stats(data, outpath):
    lines = [
        "=" * 72,
        "Phase-space and Riemann-invariant diagnostics (at t_final)",
        "=" * 72,
        f"Cell filter : h > {H_MIN_PHASE:.2e} m",
        f"Curve grid  : {N_CURVE} samples per analytical branch",
        "",
    ]
    for c in CASES:
        d = data[c["key"]]
        h_pts = d["h"][-1]
        u_pts = d["u"][-1]
        m = h_pts > H_MIN_PHASE
        cL = np.sqrt(d["g"] * h_pts[m])
        Rp = u_pts[m] + 2.0 * cL
        Rm = u_pts[m] - 2.0 * cL
        curves = analytical_curves(c["key"], d)
        dist = nearest_curve_distance(h_pts[m], u_pts[m], curves)
        lines += [
            "-" * 72,
            f"Case: {c['key']}  ({c['label']})",
            f"  active cells (h > {H_MIN_PHASE:.0e}): {int(m.sum())} "
            f"of {len(h_pts)}",
            f"  R+: min={Rp.min():+.4f}, max={Rp.max():+.4f}, "
            f"mean={Rp.mean():+.4f}, std={Rp.std():.4e}",
            f"  R-: min={Rm.min():+.4f}, max={Rm.max():+.4f}, "
            f"mean={Rm.mean():+.4f}, std={Rm.std():.4e}",
        ]
        if curves:
            lines += [
                f"  distance to nearest analytical curve in (h, u):",
                f"    median               : {np.median(dist):.6e}",
                f"    mean                 : {np.mean(dist):.6e}",
                f"    max                  : {np.max(dist):.6e}",
                f"    p95                  : "
                f"{np.percentile(dist, 95):.6e}",
            ]
        lines.append("")
    outpath.write_text("\n".join(lines))


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
