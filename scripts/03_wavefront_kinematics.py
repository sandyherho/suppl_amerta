#!/usr/bin/env python
"""Wave-front kinematics: extracted positions vs analytical celerities.

For each Riemann problem the dominant wave fronts (rarefaction edges,
shocks, dry tip) are extracted from the depth field and compared to
the analytical wave speeds derived from the Riemann star state.

Inputs : ../data/case_*.nc
Outputs: ../figs/03_wavefront_kinematics.{pdf,png,eps}
         ../stats/03_wavefront_kinematics.txt
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
OUTPUT_NAME = "03_wavefront_kinematics"

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

H_DRY_FACTOR = 5.0    # dry-tip threshold = factor x h_dry_threshold
DEV_TOL_FRAC = 0.005  # relative tolerance for "disturbed" detection
SHOCK_HALF_WIN = 80.0  # +/- m around analytical shock for argmax|dh/dx|
T_MIN_FRAC = 0.02     # discard the first 2% of timesteps for fits/celerity

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
        attrs = {k: getattr(nc, k) for k in
                 ("g", "h_left", "h_right", "u_left", "u_right",
                  "L", "h_dry_threshold")}
        attrs = {k: float(v) for k, v in attrs.items()}
        return {
            "x": np.asarray(nc.variables["x"][:]),
            "t": np.asarray(nc.variables["time"][:]),
            "h": np.asarray(nc.variables["h"][:]),
            **attrs,
        }


# ---------------------------------------------------------- analytical states
def stoker_star(hL, hR, uL, uR, g):
    cL = np.sqrt(g * hL)

    def fn(h):
        c = np.sqrt(g * h)
        rar = uL + 2.0 * (cL - c)
        sho = uR + (h - hR) * np.sqrt(g * (h + hR) / (2.0 * h * hR))
        return rar - sho

    h_star = brentq(fn, hR * 1.0001, hL * 0.9999, xtol=1e-12, maxiter=200)
    u_star = uL + 2.0 * (cL - np.sqrt(g * h_star))
    S_shock = (h_star * u_star - hR * uR) / (h_star - hR)
    return h_star, u_star, S_shock


def double_shock_star(h0, U, g):
    def fn(h):
        return U - (h - h0) * np.sqrt(g * (h + h0) / (2.0 * h * h0))

    h_star = brentq(fn, h0 * 1.0001, h0 * 500.0, xtol=1e-12, maxiter=200)
    S_L = h0 * U / (h0 - h_star)
    return h_star, S_L


# --------------------------------------------------------- front extraction
def extract_left_disturbance(x, h_row, h_baseline, tol_frac):
    """Leftmost cell where |h - h_baseline| > tol_frac * h_baseline."""
    tol = tol_frac * h_baseline
    idx = np.where(np.abs(h_row - h_baseline) > tol)[0]
    return x[idx[0]] if idx.size else np.nan


def extract_right_disturbance(x, h_row, h_baseline, tol_frac):
    """Rightmost cell where |h - h_baseline| > tol_frac * h_baseline."""
    tol = tol_frac * h_baseline
    idx = np.where(np.abs(h_row - h_baseline) > tol)[0]
    return x[idx[-1]] if idx.size else np.nan


def extract_shock_position(x, h_row, x_anchor, half_window):
    """argmax |dh/dx| within [x_anchor - hw, x_anchor + hw]."""
    grad = np.abs(np.diff(h_row))
    xc = 0.5 * (x[:-1] + x[1:])
    mask = (xc >= x_anchor - half_window) & (xc <= x_anchor + half_window)
    if not mask.any():
        return np.nan
    sub = np.where(mask)[0]
    return xc[sub[np.argmax(grad[sub])]]


def extract_dry_tip(x, h_row, h_dry):
    """Rightmost cell where h > h_dry."""
    wet = np.where(h_row > h_dry)[0]
    return x[wet[-1]] if wet.size else np.nan


# -------------------------------- per-case extraction + analytical references
def extract_fronts(case_key, d):
    """Return a dict of named fronts (each a (t, x) tuple) plus analytical
    reference lines for the figure, keyed by case."""
    x = d["x"]
    t = d["t"]
    h = d["h"]
    g = d["g"]
    L = d["L"]
    x_dam = 0.5 * L
    h_dry = H_DRY_FACTOR * d["h_dry_threshold"]
    nt = len(t)

    fronts = {}
    refs = []  # list of dicts: {label, slope, intercept, color, ls}

    if case_key == "stoker":
        cL = np.sqrt(g * d["h_left"])
        h_s, u_s, S = stoker_star(d["h_left"], d["h_right"],
                                  d["u_left"], d["u_right"], g)
        x_lhead = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        x_shock = np.array([extract_shock_position(
            x, h[i], x_dam + S * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        fronts["L-rar head"] = (t, x_lhead, "o", "C0")
        fronts["shock"] = (t, x_shock, "s", "C3")
        refs.append({"label": fr"$-c_L = {-cL:.3f}$ m/s",
                     "slope": -cL, "intercept": x_dam,
                     "color": "C0", "ls": "--"})
        refs.append({"label": fr"$S = {S:.3f}$ m/s",
                     "slope": S, "intercept": x_dam,
                     "color": "C3", "ls": "--"})

    elif case_key == "ritter":
        cL = np.sqrt(g * d["h_left"])
        x_lhead = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        x_dry = np.array([extract_dry_tip(x, h[i], h_dry)
                          for i in range(nt)])
        fronts["L-rar head"] = (t, x_lhead, "o", "C0")
        fronts["dry tip"] = (t, x_dry, "^", "C2")
        refs.append({"label": fr"$-c_L = {-cL:.3f}$ m/s",
                     "slope": -cL, "intercept": x_dam,
                     "color": "C0", "ls": "--"})
        refs.append({"label": fr"$2c_L = {2*cL:.3f}$ m/s",
                     "slope": 2.0 * cL, "intercept": x_dam,
                     "color": "C2", "ls": "--"})

    elif case_key == "double_rarefaction":
        c0 = np.sqrt(g * d["h_left"])
        uL = d["u_left"]
        uR = d["u_right"]
        x_lhead = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        x_rhead = np.array([extract_right_disturbance(
            x, h[i], d["h_right"], DEV_TOL_FRAC) for i in range(nt)])
        fronts["L-rar head"] = (t, x_lhead, "o", "C0")
        fronts["R-rar head"] = (t, x_rhead, "v", "C3")
        refs.append({"label": fr"$u_L - c_0 = {uL - c0:.3f}$ m/s",
                     "slope": uL - c0, "intercept": x_dam,
                     "color": "C0", "ls": "--"})
        refs.append({"label": fr"$u_R + c_0 = {uR + c0:.3f}$ m/s",
                     "slope": uR + c0, "intercept": x_dam,
                     "color": "C3", "ls": "--"})

    elif case_key == "double_shock":
        h_s, S_L = double_shock_star(d["h_left"], d["u_left"], g)
        x_lshock = np.array([extract_shock_position(
            x, h[i], x_dam + S_L * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        x_rshock = np.array([extract_shock_position(
            x, h[i], x_dam - S_L * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        fronts["L-shock"] = (t, x_lshock, "s", "C0")
        fronts["R-shock"] = (t, x_rshock, "D", "C3")
        refs.append({"label": fr"$S_L = {S_L:.3f}$ m/s",
                     "slope": S_L, "intercept": x_dam,
                     "color": "C0", "ls": "--"})
        refs.append({"label": fr"$-S_L = {-S_L:.3f}$ m/s",
                     "slope": -S_L, "intercept": x_dam,
                     "color": "C3", "ls": "--"})

    return fronts, refs


def primary_celerity(case_key, d):
    """Return (label, c_an) for the primary wave used in panel (e)."""
    g = d["g"]
    if case_key == "stoker":
        _, _, S = stoker_star(d["h_left"], d["h_right"],
                              d["u_left"], d["u_right"], g)
        return "shock", S
    if case_key == "ritter":
        return "dry tip", 2.0 * np.sqrt(g * d["h_left"])
    if case_key == "double_rarefaction":
        c0 = np.sqrt(g * d["h_left"])
        return "R-rar head", d["u_right"] + c0
    if case_key == "double_shock":
        _, S_L = double_shock_star(d["h_left"], d["u_left"], g)
        return "R-shock", -S_L
    return "", np.nan


def primary_position_series(case_key, d):
    fronts, _ = extract_fronts(case_key, d)
    if case_key == "stoker":
        return fronts["shock"]
    if case_key == "ritter":
        return fronts["dry tip"]
    if case_key == "double_rarefaction":
        return fronts["R-rar head"]
    if case_key == "double_shock":
        return fronts["R-shock"]
    raise ValueError(f"unknown case: {case_key}")


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(11.0, 6.5))
    gs = fig.add_gridspec(
        2, 3, left=0.07, right=0.985, top=0.97, bottom=0.10,
        hspace=0.32, wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[:, 2])]
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for ax, tag, c in zip(axes[:4], tags[:4], CASES):
        d = data[c["key"]]
        x_dam = 0.5 * d["L"]
        fronts, refs = extract_fronts(c["key"], d)
        for name, (tt, xx, marker, color) in fronts.items():
            ax.plot(tt, xx, marker=marker, ms=2.5, lw=0,
                    color=color, alpha=0.7, label=name)
        for ref in refs:
            tt = d["t"]
            ax.plot(tt, ref["intercept"] + ref["slope"] * tt,
                    color=ref["color"], ls=ref["ls"], lw=1.0,
                    label=ref["label"])
        ax.axhline(x_dam, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("x [m]")
        ax.text(0.03, 0.95, tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="best", fontsize=7, frameon=False)
        ax.grid(alpha=0.3)

    ax = axes[4]
    for c in CASES:
        d = data[c["key"]]
        _, c_an = primary_celerity(c["key"], d)
        tt, xx, _, _ = primary_position_series(c["key"], d)
        sel = (tt > tt[-1] * T_MIN_FRAC) & ~np.isnan(xx)
        if sel.sum() == 0 or c_an == 0:
            continue
        x_dam = 0.5 * d["L"]
        c_num = (xx[sel] - x_dam) / tt[sel]
        ax.plot(tt[sel], (c_num - c_an) / abs(c_an),
                color=c["color"], lw=1.1, label=c["label"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$(c_{\mathrm{num}} - c_{\mathrm{an}}) / |c_{\mathrm{an}}|$")
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
        "Wave-front kinematics",
        "=" * 72,
        f"H_dry factor                : {H_DRY_FACTOR}",
        f"Disturbance tolerance       : {DEV_TOL_FRAC * 100:.2f}% of "
        f"baseline depth",
        f"Shock-search half window    : {SHOCK_HALF_WIN:.1f} m",
        f"Linear fit skipped first    : {T_MIN_FRAC * 100:.1f}% of t",
        "",
    ]

    for c in CASES:
        d = data[c["key"]]
        x_dam = 0.5 * d["L"]
        fronts, _ = extract_fronts(c["key"], d)
        prim_name, c_an = primary_celerity(c["key"], d)
        lines += ["-" * 72,
                  f"Case: {c['key']}  ({c['label']})",
                  f"  primary wave              : {prim_name}",
                  f"  analytical primary celerity: {c_an:+.6f} m/s"]
        for name, (tt, xx, _, _) in fronts.items():
            sel = (tt > tt[-1] * T_MIN_FRAC) & ~np.isnan(xx)
            if sel.sum() < 5:
                lines.append(f"  {name:<15s}: insufficient samples")
                continue
            slope, intercept = np.polyfit(tt[sel], xx[sel], 1)
            origin_offset = intercept - x_dam
            mean_resid = float(np.mean(xx[sel] - (intercept + slope * tt[sel])))
            std_resid = float(np.std(xx[sel] - (intercept + slope * tt[sel])))
            lines += [
                f"  {name}:",
                f"    fitted speed            : {slope:+.6f} m/s",
                f"    fitted origin (x - x_dam): {origin_offset:+.6f} m",
                f"    residual mean / std     : {mean_resid:+.4e} / "
                f"{std_resid:.4e} m",
            ]
        # primary celerity stats
        tt, xx, _, _ = primary_position_series(c["key"], d)
        sel = (tt > tt[-1] * T_MIN_FRAC) & ~np.isnan(xx)
        if sel.sum() and c_an != 0:
            c_num = (xx[sel] - x_dam) / tt[sel]
            err = (c_num - c_an) / abs(c_an)
            lines += [
                f"  primary celerity error (rel.):",
                f"    final          : {err[-1]:+.6e}",
                f"    mean over t    : {err.mean():+.6e}",
                f"    std over t     : {err.std():.6e}",
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
