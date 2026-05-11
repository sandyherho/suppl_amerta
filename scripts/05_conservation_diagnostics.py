#!/usr/bin/env python
"""Conservation diagnostics with proper boundary-flux corrections.

  (a) BC-corrected mass closure residual.
  (b) Cumulative dissipation D(t) (entropy condition D >= 0).
  (c) Wet-cell maximum Froude with strict wet threshold.
  (d) TV(q)(t) - TV(q)(0); q is the conserved variable.

Inputs : ../data/case_*.nc
Outputs: ../figs/05_conservation_diagnostics.{pdf,png,eps}
         ../stats/05_conservation_diagnostics.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "05_conservation_diagnostics"

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

H_DRY_STRICT = 0.05

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
            "x":      np.asarray(nc.variables["x"][:]),
            "t":      np.asarray(nc.variables["time"][:]),
            "h":      np.asarray(nc.variables["h"][:]),
            "u":      np.asarray(nc.variables["u"][:]),
            "q":      np.asarray(nc.variables["q"][:]),
            "mass":   np.asarray(nc.variables["mass_integral"][:]),
            "energy": np.asarray(nc.variables["energy_integral"][:]),
            "g":      float(nc.g),
            "L":      float(nc.L),
        }


def style_axes(ax):
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=6)
    ax.tick_params(direction="out", length=3)
    ax.margins(x=0.02, y=0.06)
    ax.grid(alpha=0.3)


def trapz_cumulative(t, y):
    if len(t) < 2:
        return np.zeros_like(t)
    return np.concatenate(
        ([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))))


def cumulative_mass_inflow(t, q):
    return trapz_cumulative(t, q[:, 0] - q[:, -1])


def energy_flux(h, u, g):
    return u * (0.5 * u**2 * h + g * h**2)


def cumulative_energy_inflow(t, h, u, g):
    F_left  = energy_flux(h[:,  0], u[:,  0], g)
    F_right = energy_flux(h[:, -1], u[:, -1], g)
    return trapz_cumulative(t, F_left - F_right)


def mass_residual(t, mass, q):
    return (mass - mass[0]) - cumulative_mass_inflow(t, q)


def cumulative_dissipation(t, h, u, energy, g):
    return cumulative_energy_inflow(t, h, u, g) - (energy - energy[0])


def wet_max_froude(h, u, g, h_thresh):
    nt = h.shape[0]
    fr = np.full(nt, np.nan)
    for i in range(nt):
        wet = h[i] > h_thresh
        if wet.any():
            fr[i] = float(np.max(np.abs(u[i, wet]) / np.sqrt(g * h[i, wet])))
    return fr


def total_variation(field):
    return np.sum(np.abs(np.diff(field, axis=1)), axis=1)


def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.97, top=0.97, bottom=0.13,
        hspace=0.32, wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for c in CASES:
        d = data[c["key"]]
        rM = mass_residual(d["t"], d["mass"], d["q"])
        D  = cumulative_dissipation(d["t"], d["h"], d["u"],
                                    d["energy"], d["g"])
        Fr = wet_max_froude(d["h"], d["u"], d["g"], H_DRY_STRICT)
        tv_q = total_variation(d["q"])
        axes[0].plot(d["t"], rM,             color=c["color"], lw=1.0)
        axes[1].plot(d["t"], D,              color=c["color"], lw=1.0)
        axes[2].plot(d["t"], Fr,             color=c["color"], lw=1.0)
        axes[3].plot(d["t"], tv_q - tv_q[0], color=c["color"], lw=1.0)

    axes[0].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[1].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[2].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[3].axhline(0.0, color="k", lw=0.6, ls=":")

    axes[0].set_ylabel(r"$M(t)-M(0)-\!\int q_{\mathrm{BC}}\,ds$  [m$^2$]")
    axes[1].set_ylabel(r"cumulative dissipation $D(t)$  [m$^4$ s$^{-2}$]")
    axes[2].set_ylabel(rf"$\max\{{|u|/\sqrt{{gh}} : h>{H_DRY_STRICT}\,\mathrm{{m}}\}}$")
    axes[3].set_ylabel(r"$\mathrm{TV}(q)(t)-\mathrm{TV}(q)(0)$  [m$^2$ s$^{-1}$]")

    # cleaner scientific notation for the very small mass residual axis
    fmt_sci = mticker.ScalarFormatter(useMathText=True)
    fmt_sci.set_powerlimits((-2, 3))
    axes[0].yaxis.set_major_formatter(fmt_sci)

    for ax, tag in zip(axes, tags):
        ax.set_xlabel(r"$t$  [s]")
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
        "  AMERTA  --  Conservation diagnostics with BC corrections",
        sep,
        "",
        "  Mass         residual_M(t) = M(t)-M(0) - integral(q_L - q_R) ds",
        "  Energy flux  F_E = u (1/2 u^2 h + g h^2)",
        "  Dissipation  D(t) = integral(F_E_L - F_E_R) ds - (E(t) - E(0))",
        "  Wet Froude   max(|u|/sqrt(gh)) over cells h > strict floor",
        "  TVD probe    TV(q)(t) - TV(q)(0); q is the conserved variable",
        "",
        "  Note  E(t) is the solver-stored energy_integral.  Its physical",
        "        unit is m^4/s^2 (depth-averaged specific energy density",
        "        m^3/s^2 integrated over channel length m); the NetCDF",
        "        attribute label of m^3/s^2 is dimensionally inconsistent.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Strict wet-cell threshold",
            H_DRY_STRICT, "m  (5 x solver H_DRY = 0.01)"),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        rM = mass_residual(d["t"], d["mass"], d["q"])
        E_in = cumulative_energy_inflow(d["t"], d["h"], d["u"], d["g"])
        D = E_in - (d["energy"] - d["energy"][0])
        Fr = wet_max_froude(d["h"], d["u"], d["g"], H_DRY_STRICT)
        tv_q = total_variation(d["q"])
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["x"].size)),
            "",
            "    -- mass closure (BC-corrected) --",
            fmt("M(0)",   float(d["mass"][0]),  "m^2"),
            fmt("M(t_f)", float(d["mass"][-1]), "m^2"),
            fmt("cumulative net mass inflow at t_f",
                float(cumulative_mass_inflow(d["t"], d["q"])[-1]), "m^2"),
            fmt("max |residual|",      float(np.max(np.abs(rM))), "m^2"),
            fmt("max |residual|/M(0)",
                float(np.max(np.abs(rM)) / d["mass"][0]), ""),
            "",
            "    -- energy budget (BC-corrected) --",
            fmt("E(0)",   float(d["energy"][0]),  "m^4/s^2"),
            fmt("E(t_f)", float(d["energy"][-1]), "m^4/s^2"),
            fmt("cumulative net E inflow at t_f",
                float(E_in[-1]), "m^4/s^2"),
            fmt("cumulative dissipation D(t_f)",
                float(D[-1]), "m^4/s^2"),
            fmt("min D(t)  (must be >= 0)",
                float(np.nanmin(D)), "m^4/s^2"),
            "",
            "    -- wet-cell Froude (strict mask) --",
            fmt("peak max-Froude over t",
                float(np.nanmax(Fr)), ""),
            fmt("supercritical fraction of t",
                float(np.nansum(Fr > 1.0) / len(Fr)), ""),
            "",
            "    -- TV(q) (proper TVD probe) --",
            fmt("TV(q)(0)",   float(tv_q[0]),  "m^2/s"),
            fmt("TV(q)(t_f)", float(tv_q[-1]), "m^2/s"),
            fmt("max TV(q) growth above TV(q)(0)",
                float(np.max(tv_q - tv_q[0])), "m^2/s"),
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
