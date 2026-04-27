#!/usr/bin/env python
"""Error localization in space and temporal-scaling fingerprint.

Each panel is a single cross-case overlay of all four Riemann
problems:

  (a) pointwise depth error |h_num - h_an|(x) at the final time,
  (b) empirical CDF of pointwise depth error at the final time,
  (c) L1(h)(t) on log-log axes with fitted power law t^alpha,
  (d) L1(q)(t) on log-log axes with fitted power law t^alpha.

Shock-front FWHM diagnostics for cases 1 and 4 are reported in the
stats file rather than the figure.

Inputs : ../data/case_*.nc
Outputs: ../figs/05_error_localization_temporal_scaling.{pdf,png,eps}
         ../stats/05_error_localization_temporal_scaling.txt
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
OUTPUT_NAME = "05_error_localization_temporal_scaling"

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

T_FIT_MIN_FRAC = 0.10  # discard first 10% of t when fitting power laws
SHOCK_KEYS = ("stoker", "double_shock")
ERR_PERCENTILES = (50, 75, 90, 95, 99)

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
            "h_an":   np.asarray(nc.variables["h_analytical"][:]),
            "h_err":  np.asarray(nc.variables["h_error"][:]),
            "l1_h":   np.asarray(nc.variables["l1_h"][:]),
            "l2_h":   np.asarray(nc.variables["l2_h"][:]),
            "l1_q":   np.asarray(nc.variables["l1_q"][:]),
            "L":      float(nc.L),
        }


# ------------------------------------------------------------------- analysis
def fit_power_law(t, y, t_min_frac):
    sel = (t > t[-1] * t_min_frac) & (y > 0)
    if sel.sum() < 5:
        return np.nan, np.nan
    p = np.polyfit(np.log(t[sel]), np.log(y[sel]), 1)
    return float(p[0]), float(np.exp(p[1]))


def shock_fwhm_cells(h_row):
    g = np.abs(np.diff(h_row))
    if g.size == 0 or g.max() <= 0:
        return np.nan
    above = np.where(g >= 0.5 * g.max())[0]
    if above.size == 0:
        return np.nan
    return float(above[-1] - above[0] + 1)


def shock_width_series(h):
    return np.array([shock_fwhm_cells(h[i]) for i in range(h.shape[0])])


def empirical_cdf(values):
    v = np.sort(np.asarray(values))
    v = v[v > 0]
    if v.size == 0:
        return np.array([]), np.array([])
    p = np.arange(1, v.size + 1) / v.size
    return v, p


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

    # (a) |h_err|(x) at t_final
    ax = axes[0]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["x"], np.abs(d["h_err"][-1]), color=c["color"])
    ax.set_xlabel(r"$x$  [m]")
    ax.set_ylabel(r"$|h_{\mathrm{num}} - h_{\mathrm{an}}|$  at  $t_f$  [m]")
    ax.set_yscale("log")

    # (b) ECDF of pointwise error
    ax = axes[1]
    for c in CASES:
        d = data[c["key"]]
        v, p = empirical_cdf(np.abs(d["h_err"][-1]))
        if v.size:
            ax.plot(v, p, color=c["color"])
    ax.set_xlabel(r"$|h_{\mathrm{num}} - h_{\mathrm{an}}|$  [m]")
    ax.set_ylabel(r"empirical CDF")
    ax.set_xscale("log")

    # (c) L1(h) vs t with fitted power law
    ax = axes[2]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["l1_h"], color=c["color"])
        alpha, A = fit_power_law(d["t"], d["l1_h"], T_FIT_MIN_FRAC)
        if np.isfinite(alpha):
            tt = d["t"][d["t"] > d["t"][-1] * T_FIT_MIN_FRAC]
            ax.plot(tt, A * tt ** alpha,
                    color=c["color"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$L^1(h)$  [m]")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # (d) L1(q) vs t with fitted power law
    ax = axes[3]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["l1_q"], color=c["color"])
        alpha, A = fit_power_law(d["t"], d["l1_q"], T_FIT_MIN_FRAC)
        if np.isfinite(alpha):
            tt = d["t"][d["t"] > d["t"][-1] * T_FIT_MIN_FRAC]
            ax.plot(tt, A * tt ** alpha,
                    color=c["color"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel(r"$t$  [s]")
    ax.set_ylabel(r"$L^1(q)$  [m$^2$ s$^{-1}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for ax, tag in zip(axes, tags):
        ax.set_title(tag, fontsize=11, fontweight="bold", pad=6)
        ax.grid(alpha=0.3, which="both")

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
        "  AMERTA v0.0.3  --  Error localization and temporal scaling",
        sep,
        "",
        "  Diagnostic norms reported here:",
        "    L1(h)(t) = integral_x |h_num - h_an| dx",
        "    L2(h)(t) = sqrt(integral_x (h_num - h_an)^2 dx)",
        "    L1(q)(t) = integral_x |q_num - q_an| dx",
        "  Power-law fits use the model L_p(t) = A * t^alpha.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Initial fraction of t skipped in fits",
            T_FIT_MIN_FRAC * 100.0, "% of t_final"),
        fmt("Shock width metric", "FWHM of |dh/dx|"),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        err_final = np.abs(d["h_err"][-1])
        a_l1, A_l1 = fit_power_law(d["t"], d["l1_h"], T_FIT_MIN_FRAC)
        a_l2, A_l2 = fit_power_law(d["t"], d["l2_h"], T_FIT_MIN_FRAC)
        a_q, A_q = fit_power_law(d["t"], d["l1_q"], T_FIT_MIN_FRAC)

        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            "",
            "    -- final-time norms --",
            fmt("L1(h) at t_f", float(d["l1_h"][-1]), "m"),
            fmt("L2(h) at t_f", float(d["l2_h"][-1]), "m"),
            fmt("L1(q) at t_f", float(d["l1_q"][-1]), "m^2 s^-1"),
            "",
            "    -- power-law scaling L_p(t) ~ A * t^alpha --",
            fmt("L1(h) : alpha", a_l1, ""),
            fmt("L1(h) : A", A_l1, "m s^-alpha"),
            fmt("L2(h) : alpha", a_l2, ""),
            fmt("L2(h) : A", A_l2, "m s^-alpha"),
            fmt("L1(q) : alpha", a_q, ""),
            fmt("L1(q) : A", A_q, "m^2 s^(-1-alpha)"),
            "",
            "    -- pointwise |h_err| at t_f --",
            fmt("max", float(err_final.max()), "m"),
            fmt("mean", float(err_final.mean()), "m"),
        ]
        for p in ERR_PERCENTILES:
            lines.append(fmt(f"{p:2d}th percentile",
                             float(np.percentile(err_final, p)), "m"))
        if c["key"] in SHOCK_KEYS:
            w = shock_width_series(d["h"])
            wf = w[~np.isnan(w)]
            if wf.size:
                lines += [
                    "",
                    "    -- shock-front width FWHM(|dh/dx|) --",
                    fmt("at t_f", float(w[-1]), "cells"),
                    fmt("mean over t", float(wf.mean()), "cells"),
                    fmt("std over t", float(wf.std()), "cells"),
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
