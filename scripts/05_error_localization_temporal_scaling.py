#!/usr/bin/env python
"""Error localization in space and temporal-scaling fingerprint.

For each Riemann problem the script visualises:

  - where in x the depth error concentrates at the final time,
  - the empirical CDF of pointwise depth error,
  - the temporal growth of L1(h) and L2(h) on log-log axes (with a
    fitted power law L_p(t) ~ t^alpha),
  - the temporal growth of the discharge norm L1(q) (the v0.0.4
    momentum metric well-suited to the dry-bed Ritter case),
  - the shock-front width (FWHM of |dh/dx|) for the two cases that
    contain shocks (1 and 4).

Inputs : ../data/case_*.nc
Outputs: ../figs/05_error_localization_temporal_scaling.{pdf,png,eps}
         ../stats/05_error_localization_temporal_scaling.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

T_FIT_MIN_FRAC = 0.10  # skip the first 10% of t when fitting power laws
SHOCK_KEYS = ("stoker", "double_shock")

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
    """Fit y(t) ~ A * t^alpha by least squares in log space."""
    sel = (t > t[-1] * t_min_frac) & (y > 0)
    if sel.sum() < 5:
        return np.nan, np.nan
    p = np.polyfit(np.log(t[sel]), np.log(y[sel]), 1)
    return float(p[0]), float(np.exp(p[1]))


def shock_fwhm_cells(x, h_row):
    """FWHM of |dh/dx| in number of cells; NaN if signal is flat."""
    g = np.abs(np.diff(h_row))
    if g.size == 0:
        return np.nan
    gmax = g.max()
    if gmax <= 0:
        return np.nan
    above = np.where(g >= 0.5 * gmax)[0]
    if above.size == 0:
        return np.nan
    return float(above[-1] - above[0] + 1)


def shock_width_series(x, h):
    return np.array([shock_fwhm_cells(x, h[i]) for i in range(h.shape[0])])


def empirical_cdf(values):
    v = np.sort(np.asarray(values))
    v = v[v > 0]
    if v.size == 0:
        return np.array([]), np.array([])
    p = np.arange(1, v.size + 1) / v.size
    return v, p


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

    # (a) |h - h_an|(x) at t_final
    ax = axes[0]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["x"], np.abs(d["h_err"][-1]),
                color=c["color"], label=c["label"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$|h_{\mathrm{num}} - h_{\mathrm{an}}|$  at  $t_f$  [m]")
    ax.set_yscale("log")
    ax.text(0.03, 0.95, tags[0], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    # (b) ECDF of pointwise error at t_final
    ax = axes[1]
    for c in CASES:
        d = data[c["key"]]
        v, p = empirical_cdf(np.abs(d["h_err"][-1]))
        if v.size:
            ax.plot(v, p, color=c["color"], label=c["label"])
    ax.set_xlabel(r"$|h_{\mathrm{num}} - h_{\mathrm{an}}|$  [m]")
    ax.set_ylabel(r"empirical CDF")
    ax.set_xscale("log")
    ax.text(0.03, 0.95, tags[1], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    # (c) L1(h) and L2(h) vs t, log-log + fitted power law
    ax = axes[2]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["l1_h"], color=c["color"], label=c["label"])
        alpha, A = fit_power_law(d["t"], d["l1_h"], T_FIT_MIN_FRAC)
        if np.isfinite(alpha):
            tt = d["t"][d["t"] > d["t"][-1] * T_FIT_MIN_FRAC]
            ax.plot(tt, A * tt ** alpha,
                    color=c["color"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$L^1(h)$ [m]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.03, 0.95, tags[2], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    # (d) L1(q) vs t, log-log
    ax = axes[3]
    for c in CASES:
        d = data[c["key"]]
        ax.plot(d["t"], d["l1_q"], color=c["color"], label=c["label"])
        alpha, A = fit_power_law(d["t"], d["l1_q"], T_FIT_MIN_FRAC)
        if np.isfinite(alpha):
            tt = d["t"][d["t"] > d["t"][-1] * T_FIT_MIN_FRAC]
            ax.plot(tt, A * tt ** alpha,
                    color=c["color"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$L^1(q)$ [m$^2$ s$^{-1}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.03, 0.95, tags[3], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    # (e) shock-width FWHM vs t for cases with shocks
    ax = axes[4]
    for c in CASES:
        if c["key"] not in SHOCK_KEYS:
            continue
        d = data[c["key"]]
        w = shock_width_series(d["x"], d["h"])
        ax.plot(d["t"], w, color=c["color"], label=c["label"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"shock FWHM [cells]")
    ax.text(0.03, 0.95, tags[4], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------- stats
def write_stats(data, outpath):
    lines = [
        "=" * 72,
        "Error localization and temporal-scaling fingerprint",
        "=" * 72,
        f"Power-law fit window : t > {T_FIT_MIN_FRAC * 100:.1f}% of t_final",
        "Shock width metric   : FWHM of |dh/dx| in cells",
        "",
    ]
    pcts = (50, 75, 90, 95, 99)
    for c in CASES:
        d = data[c["key"]]
        err_final = np.abs(d["h_err"][-1])
        a_l1, A_l1 = fit_power_law(d["t"], d["l1_h"], T_FIT_MIN_FRAC)
        a_l2, A_l2 = fit_power_law(d["t"], d["l2_h"], T_FIT_MIN_FRAC)
        a_q, A_q = fit_power_law(d["t"], d["l1_q"], T_FIT_MIN_FRAC)

        lines += [
            "-" * 72,
            f"Case: {c['key']}  ({c['label']})",
            f"  L1(h) at t_final          : {d['l1_h'][-1]:.6e} m",
            f"  L2(h) at t_final          : {d['l2_h'][-1]:.6e} m",
            f"  L1(q) at t_final          : {d['l1_q'][-1]:.6e} m^2/s",
            f"  L1(h) ~ A * t^alpha       : alpha={a_l1:+.4f}, A={A_l1:.4e}",
            f"  L2(h) ~ A * t^alpha       : alpha={a_l2:+.4f}, A={A_l2:.4e}",
            f"  L1(q) ~ A * t^alpha       : alpha={a_q:+.4f}, A={A_q:.4e}",
            f"  pointwise |h_err| at t_f  :",
            f"    max                     : {err_final.max():.6e} m",
            f"    mean                    : {err_final.mean():.6e} m",
        ]
        for p in pcts:
            lines.append(
                f"    {p:2d}th percentile        : "
                f"{np.percentile(err_final, p):.6e} m"
            )
        if c["key"] in SHOCK_KEYS:
            w = shock_width_series(d["x"], d["h"])
            wf = w[~np.isnan(w)]
            if wf.size:
                lines += [
                    f"  shock FWHM at t_final     : {w[-1]:.2f} cells",
                    f"  shock FWHM mean over t    : {wf.mean():.2f} cells",
                    f"  shock FWHM std over t     : {wf.std():.2f} cells",
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
