#!/usr/bin/env python
"""Conservation invariants, TVD property, and symmetry preservation.

Verifies the discrete solution against the four physical invariants of
the homogeneous shallow-water Riemann problem under the simulator's
Neumann (zero-gradient) boundary conditions:

  - mass closure including BC throughflow,
  - non-negative shock-only energy dissipation rate,
  - non-increasing total variation of h (TVD property),
  - center-of-mass invariance for symmetric ICs.

The peak Froude number trajectory is included as a flow-regime probe.

Inputs : ../data/case_*.nc
Outputs: ../figs/04_invariants_tvd_symmetry.{pdf,png,eps}
         ../stats/04_invariants_tvd_symmetry.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


# --------------------------------------------------------------- configuration
DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "04_invariants_tvd_symmetry"

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

SYMMETRIC_KEYS = ("double_rarefaction", "double_shock")

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
        return {
            "x":         np.asarray(nc.variables["x"][:]),
            "t":         np.asarray(nc.variables["time"][:]),
            "h":         np.asarray(nc.variables["h"][:]),
            "q":         np.asarray(nc.variables["q"][:]),
            "mass":      np.asarray(nc.variables["mass_integral"][:]),
            "energy":    np.asarray(nc.variables["energy_integral"][:]),
            "froude":    np.asarray(nc.variables["froude_max"][:]),
            "L":         float(nc.L),
        }


# ------------------------------------------------------------------- analysis
def cumulative_bc_flux(t, q):
    """Cumulative net mass that has crossed the boundaries (m^2).

    Net inflow = q[:, 0] - q[:, -1]   (positive = mass entering domain).
    Integrated by the trapezoid rule.
    """
    q_in = q[:, 0] - q[:, -1]
    return np.concatenate(([0.0], np.cumsum(0.5 * (q_in[1:] + q_in[:-1])
                                            * np.diff(t))))


def total_variation(h):
    """TV(h)(t) = sum_i |h_{i+1} - h_i|."""
    return np.sum(np.abs(np.diff(h, axis=1)), axis=1)


def center_of_mass(x, h, dx):
    """<x>(t) = sum(x * h) * dx / sum(h * dx)."""
    num = np.sum(h * x[None, :], axis=1) * dx
    den = np.sum(h, axis=1) * dx
    return num / den


def energy_rate(t, e):
    """dE/dt via central differences (np.gradient)."""
    return np.gradient(e, t)


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(
        2, 3, left=0.07, right=0.985, top=0.97, bottom=0.08,
        hspace=0.32, wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # (a) BC-corrected mass closure
    ax = axes[0]
    for c in CASES:
        d = data[c["key"]]
        flux = cumulative_bc_flux(d["t"], d["q"])
        residual = (d["mass"] - d["mass"][0]) - flux
        ax.plot(d["t"], residual, color=c["color"], label=c["label"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$M(t)-M(0)-\!\int q_{\mathrm{BC}}\,ds$  [m$^2$]")
    ax.text(0.03, 0.95, tags[0], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    # (b) energy dissipation rate
    ax = axes[1]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], energy_rate(d["t"], d["energy"]),
                color=c["color"], label=c["label"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$dE/dt$ [m$^3$ s$^{-3}$]")
    ax.text(0.03, 0.95, tags[1], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    # (c) total variation
    ax = axes[2]
    for c in CASES:
        d = data[c["key"]]
        tv = total_variation(d["h"])
        ax.plot(d["t"], tv - tv[0], color=c["color"], label=c["label"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\mathrm{TV}(h)(t) - \mathrm{TV}(h)(0)$ [m]")
    ax.text(0.03, 0.95, tags[2], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    # (d) center of mass for symmetric cases
    ax = axes[3]
    for c in CASES:
        if c["key"] not in SYMMETRIC_KEYS:
            continue
        d = data[c["key"]]
        dx = d["x"][1] - d["x"][0]
        com = center_of_mass(d["x"], d["h"], dx)
        ax.plot(d["t"], com - 0.5 * d["L"],
                color=c["color"], label=c["label"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\langle x \rangle(t) - L/2$ [m]")
    ax.text(0.03, 0.95, tags[3], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    # (e) max Froude
    ax = axes[4]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["froude"], color=c["color"], label=c["label"])
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\max_x \mathrm{Fr}(t)$")
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
        "Invariants, TVD property, and symmetry diagnostics",
        "=" * 72,
        "Mass closure metric : residual = M(t) - M(0) - integral(q[0]-q[-1]) "
        "dt",
        "TVD metric          : TV(h)(t) = sum_i |h_{i+1} - h_i|",
        "Symmetry probe      : <x>(t) for symmetric ICs (cases 3, 4)",
        "",
    ]
    for c in CASES:
        d = data[c["key"]]
        flux = cumulative_bc_flux(d["t"], d["q"])
        residual = (d["mass"] - d["mass"][0]) - flux
        de_dt = energy_rate(d["t"], d["energy"])
        tv = total_variation(d["h"])
        dx = d["x"][1] - d["x"][0]

        lines += [
            "-" * 72,
            f"Case: {c['key']}  ({c['label']})",
            f"  M(0)                       : {d['mass'][0]:.6e} m^2",
            f"  M(t_final)                 : {d['mass'][-1]:.6e} m^2",
            f"  cumulative BC flux at end  : {flux[-1]:+.6e} m^2",
            f"  mass-closure residual max  : {np.max(np.abs(residual)):.6e}"
            " m^2",
            f"  mass-closure / M(0)        : "
            f"{np.max(np.abs(residual)) / d['mass'][0]:.6e}",
            f"  E(0)                       : {d['energy'][0]:.6e} m^3/s^2",
            f"  E(t_final)                 : {d['energy'][-1]:.6e} m^3/s^2",
            f"  energy dissipated          : "
            f"{(d['energy'][0] - d['energy'][-1]):.6e} m^3/s^2",
            f"  min dE/dt                  : {de_dt.min():+.6e}",
            f"  max dE/dt                  : {de_dt.max():+.6e}",
            f"  TV(h)(0)                   : {tv[0]:.6e} m",
            f"  TV(h)(t_final)             : {tv[-1]:.6e} m",
            f"  max TV(h)(t) - TV(h)(0)    : {(tv - tv[0]).max():+.6e} m",
            f"  peak Froude                : {np.nanmax(d['froude']):.6f}",
            f"  Froude > 1 fraction of t   : "
            f"{(d['froude'] > 1.0).sum() / len(d['froude']):.4f}",
        ]
        if c["key"] in SYMMETRIC_KEYS:
            com = center_of_mass(d["x"], d["h"], dx)
            dev = com - 0.5 * d["L"]
            lines += [
                f"  <x>(0) - L/2               : {dev[0]:+.6e} m",
                f"  max |<x> - L/2|            : {np.max(np.abs(dev)):.6e} m",
                f"  std(<x> - L/2)             : {np.std(dev):.6e} m",
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
