#!/usr/bin/env python
"""Conservation invariants and TVD property of the discrete solution.

Each panel overlays all four Riemann problems for a single diagnostic:

  (a) BC-corrected mass closure residual,
  (b) total energy dissipation rate dE/dt,
  (c) total variation of h relative to its initial value,
  (d) maximum Froude number versus time.

The center-of-mass symmetry probe (cases 3 and 4 only) is reported in
the stats file rather than in the figure, since it does not apply to
the asymmetric cases 1 and 2.

Inputs : ../data/case_*.nc
Outputs: ../figs/04_invariants_tvd_symmetry.{pdf,png,eps}
         ../stats/04_invariants_tvd_symmetry.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
            "x":      np.asarray(nc.variables["x"][:]),
            "t":      np.asarray(nc.variables["time"][:]),
            "h":      np.asarray(nc.variables["h"][:]),
            "q":      np.asarray(nc.variables["q"][:]),
            "mass":   np.asarray(nc.variables["mass_integral"][:]),
            "energy": np.asarray(nc.variables["energy_integral"][:]),
            "froude": np.asarray(nc.variables["froude_max"][:]),
            "L":      float(nc.L),
        }


# ------------------------------------------------------------------- analysis
def cumulative_bc_flux(t, q):
    """Cumulative net mass that crossed the boundaries (m^2)."""
    q_in = q[:, 0] - q[:, -1]
    return np.concatenate(
        ([0.0], np.cumsum(0.5 * (q_in[1:] + q_in[:-1]) * np.diff(t))))


def total_variation(h):
    return np.sum(np.abs(np.diff(h, axis=1)), axis=1)


def center_of_mass(x, h, dx):
    num = np.sum(h * x[None, :], axis=1) * dx
    den = np.sum(h, axis=1) * dx
    return num / den


def energy_rate(t, e):
    return np.gradient(e, t)


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.97, top=0.94, bottom=0.13,
        hspace=0.42, wspace=0.32,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    # (a) BC-corrected mass closure
    ax = axes[0]
    for c in CASES:
        d = data[c["key"]]
        flux = cumulative_bc_flux(d["t"], d["q"])
        residual = (d["mass"] - d["mass"][0]) - flux
        ax.plot(d["t"], residual, color=c["color"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$M(t)-M(0)-\!\int q_{\mathrm{BC}}\,ds$  [m$^2$]")

    # (b) energy dissipation rate
    ax = axes[1]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], energy_rate(d["t"], d["energy"]), color=c["color"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$dE/dt$  [m$^3$ s$^{-3}$]")

    # (c) total variation
    ax = axes[2]
    for c in CASES:
        d = data[c["key"]]
        tv = total_variation(d["h"])
        ax.plot(d["t"], tv - tv[0], color=c["color"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$\mathrm{TV}(h)(t) - \mathrm{TV}(h)(0)$  [m]")

    # (d) max Froude
    ax = axes[3]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["froude"], color=c["color"])
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$\max_x \mathrm{Fr}(t)$")

    for ax, tag in zip(axes, tags):
        ax.set_title(tag, fontsize=11, fontweight="bold", pad=6)
        ax.grid(alpha=0.3)

    proxies = [Line2D([0], [0], color=c["color"], lw=2.2, label=c["label"])
               for c in CASES]
    fig.legend(handles=proxies, loc="lower center", ncol=4,
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
        "  AMERTA v0.0.3  --  Invariants, TVD property, and symmetry",
        sep,
        "",
        "  Mass closure  : residual = M(t) - M(0) - integral_0^t (q_0 - q_N) ds",
        "  TVD property  : TV(h)(t) = sum_i |h_{i+1} - h_i| should not grow",
        "  Symmetry probe: <x>(t) - L/2 must remain near zero for cases 3 & 4",
        "",
        sub,
        "  PER-CASE DIAGNOSTICS",
        sub,
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        flux = cumulative_bc_flux(d["t"], d["q"])
        residual = (d["mass"] - d["mass"][0]) - flux
        de_dt = energy_rate(d["t"], d["energy"])
        tv = total_variation(d["h"])
        dx = d["x"][1] - d["x"][0]

        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            "",
            "    -- mass conservation --",
            fmt("M(0)", float(d["mass"][0]), "m^2"),
            fmt("M(t_f)", float(d["mass"][-1]), "m^2"),
            fmt("cumulative BC flux at t_f",
                float(flux[-1]), "m^2"),
            fmt("max |residual|",
                float(np.max(np.abs(residual))), "m^2"),
            fmt("max |residual| / M(0)",
                float(np.max(np.abs(residual)) / d["mass"][0]), ""),
            "",
            "    -- energy dissipation --",
            fmt("E(0)", float(d["energy"][0]), "m^3 s^-2"),
            fmt("E(t_f)", float(d["energy"][-1]), "m^3 s^-2"),
            fmt("E dissipated",
                float(d["energy"][0] - d["energy"][-1]), "m^3 s^-2"),
            fmt("min dE/dt", float(de_dt.min()), "m^3 s^-3"),
            fmt("max dE/dt", float(de_dt.max()), "m^3 s^-3"),
            "",
            "    -- total variation --",
            fmt("TV(h)(0)", float(tv[0]), "m"),
            fmt("TV(h)(t_f)", float(tv[-1]), "m"),
            fmt("max TV(h) growth above TV(h)(0)",
                float((tv - tv[0]).max()), "m"),
            "",
            "    -- Froude number --",
            fmt("peak max-Froude over t",
                float(np.nanmax(d["froude"])), ""),
            fmt("supercritical fraction of t",
                float((d["froude"] > 1.0).sum() / len(d["froude"])), ""),
        ]
        if c["key"] in SYMMETRIC_KEYS:
            com = center_of_mass(d["x"], d["h"], dx)
            dev = com - 0.5 * d["L"]
            lines += [
                "",
                "    -- symmetry probe (symmetric IC only) --",
                fmt("<x>(0) - L/2", float(dev[0]), "m"),
                fmt("max |<x>(t) - L/2|",
                    float(np.max(np.abs(dev))), "m"),
                fmt("std(<x>(t) - L/2)", float(np.std(dev)), "m"),
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
