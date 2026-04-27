#!/usr/bin/env python
"""Wave-front kinematics: extracted positions vs analytical celerities.

Each panel shows the dominant numerical wave-front trajectory for one
Riemann problem (markers, case colour) overlaid on the analytical
characteristic line (black dashed).  Stats text reports fits for both
the dominant and secondary wave fronts of every case.

Inputs : ../data/case_*.nc
Outputs: ../figs/03_wavefront_kinematics.{pdf,png,eps}
         ../stats/03_wavefront_kinematics.txt
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

H_DRY_FACTOR = 5.0     # dry-tip threshold = factor x h_dry_threshold
DEV_TOL_FRAC = 0.005   # disturbance detection tolerance, frac of baseline h
SHOCK_HALF_WIN = 80.0  # +/- m around analytical shock for argmax|dh/dx|
T_MIN_FRAC = 0.02      # fraction of timesteps skipped in linear fit / error

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
                 ("g", "h_left", "h_right", "u_left", "u_right",
                  "L", "h_dry_threshold")}
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
    tol = tol_frac * h_baseline
    idx = np.where(np.abs(h_row - h_baseline) > tol)[0]
    return x[idx[0]] if idx.size else np.nan


def extract_right_disturbance(x, h_row, h_baseline, tol_frac):
    tol = tol_frac * h_baseline
    idx = np.where(np.abs(h_row - h_baseline) > tol)[0]
    return x[idx[-1]] if idx.size else np.nan


def extract_shock_position(x, h_row, x_anchor, half_window):
    grad = np.abs(np.diff(h_row))
    xc = 0.5 * (x[:-1] + x[1:])
    mask = (xc >= x_anchor - half_window) & (xc <= x_anchor + half_window)
    if not mask.any():
        return np.nan
    sub = np.where(mask)[0]
    return xc[sub[np.argmax(grad[sub])]]


def extract_dry_tip(x, h_row, h_dry):
    wet = np.where(h_row > h_dry)[0]
    return x[wet[-1]] if wet.size else np.nan


# ---------------------------------------------- per-case wave-front catalogue
def build_fronts(case_key, d):
    """Return {name: (t, x_arr)} and primary wave key plus analytical refs."""
    x, t, h = d["x"], d["t"], d["h"]
    g, L = d["g"], d["L"]
    x_dam = 0.5 * L
    h_dry = H_DRY_FACTOR * d["h_dry_threshold"]
    nt = len(t)
    fronts = {}
    refs = {}    # {name: (slope, intercept)} for analytical reference lines
    primary = None

    if case_key == "stoker":
        cL = np.sqrt(g * d["h_left"])
        _, _, S = stoker_star(d["h_left"], d["h_right"],
                              d["u_left"], d["u_right"], g)
        fronts["L-rar head"] = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        fronts["shock"] = np.array([extract_shock_position(
            x, h[i], x_dam + S * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        refs["L-rar head"] = (-cL, x_dam)
        refs["shock"] = (S, x_dam)
        primary = "shock"

    elif case_key == "ritter":
        cL = np.sqrt(g * d["h_left"])
        fronts["L-rar head"] = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        fronts["dry tip"] = np.array(
            [extract_dry_tip(x, h[i], h_dry) for i in range(nt)])
        refs["L-rar head"] = (-cL, x_dam)
        refs["dry tip"] = (2.0 * cL, x_dam)
        primary = "dry tip"

    elif case_key == "double_rarefaction":
        c0 = np.sqrt(g * d["h_left"])
        uL = d["u_left"]
        uR = d["u_right"]
        fronts["L-rar head"] = np.array([extract_left_disturbance(
            x, h[i], d["h_left"], DEV_TOL_FRAC) for i in range(nt)])
        fronts["R-rar head"] = np.array([extract_right_disturbance(
            x, h[i], d["h_right"], DEV_TOL_FRAC) for i in range(nt)])
        refs["L-rar head"] = (uL - c0, x_dam)
        refs["R-rar head"] = (uR + c0, x_dam)
        primary = "R-rar head"

    elif case_key == "double_shock":
        _, S_L = double_shock_star(d["h_left"], d["u_left"], g)
        fronts["L-shock"] = np.array([extract_shock_position(
            x, h[i], x_dam + S_L * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        fronts["R-shock"] = np.array([extract_shock_position(
            x, h[i], x_dam - S_L * t[i], SHOCK_HALF_WIN) for i in range(nt)])
        refs["L-shock"] = (S_L, x_dam)
        refs["R-shock"] = (-S_L, x_dam)
        primary = "R-shock"

    return fronts, refs, primary


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.94, bottom=0.13,
        hspace=0.42, wspace=0.28,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        d = data[c["key"]]
        fronts, refs, primary = build_fronts(c["key"], d)

        x_prim = fronts[primary]
        slope, intercept = refs[primary]

        valid = ~np.isnan(x_prim)
        ax.plot(d["t"][valid], x_prim[valid], "o", ms=3.0,
                mfc=c["color"], mec=c["color"], mew=0.6, alpha=0.85)
        ax.plot(d["t"], intercept + slope * d["t"],
                color="k", ls="--", lw=1.1, alpha=0.85)
        ax.axhline(0.5 * d["L"], color="gray", ls=":", lw=0.6, alpha=0.6)

        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(r"$x$  [m]")
        ax.set_title(tag, fontsize=11, fontweight="bold", pad=6)
        ax.grid(alpha=0.3)

    proxies = [
        Line2D([0], [0], marker="o", color=CASES[0]["color"], lw=0,
               mfc=CASES[0]["color"], ms=6, label=CASES[0]["label"]),
        Line2D([0], [0], marker="o", color=CASES[1]["color"], lw=0,
               mfc=CASES[1]["color"], ms=6, label=CASES[1]["label"]),
        Line2D([0], [0], marker="o", color=CASES[2]["color"], lw=0,
               mfc=CASES[2]["color"], ms=6, label=CASES[2]["label"]),
        Line2D([0], [0], marker="o", color=CASES[3]["color"], lw=0,
               mfc=CASES[3]["color"], ms=6, label=CASES[3]["label"]),
        Line2D([0], [0], color="k", ls="--", lw=1.4, label="Analytical"),
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


def linear_fit(t, x, t_min_frac):
    """Return (slope, intercept, residual_std) skipping early samples."""
    valid = ~np.isnan(x)
    valid[: int(t_min_frac * len(t))] = False
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(t[valid], x[valid], 1)
    resid = x[valid] - (intercept + slope * t[valid])
    return float(slope), float(intercept), float(resid.std())


def write_stats(data, outpath):
    sep = "=" * 80
    sub = "-" * 80
    lines = [
        sep,
        "  AMERTA v0.0.3  --  Wave-front kinematics",
        sep,
        "",
        "  For each case the dominant and secondary wave fronts are",
        "  extracted from the simulated h field and compared against",
        "  the corresponding analytical characteristic / shock speeds.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Dry-tip threshold factor", H_DRY_FACTOR,
            "x h_dry_threshold"),
        fmt("Disturbance detection tolerance",
            DEV_TOL_FRAC * 100.0, "% of baseline depth"),
        fmt("Shock-search half window", SHOCK_HALF_WIN, "m"),
        fmt("Initial fraction of t skipped",
            T_MIN_FRAC * 100.0, "% of t_final"),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        x_dam = 0.5 * d["L"]
        fronts, refs, primary = build_fronts(c["key"], d)
        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            fmt("Primary wave (in figure)", primary),
            "",
        ]
        for name, x_arr in fronts.items():
            slope, intercept, resid = linear_fit(d["t"], x_arr, T_MIN_FRAC)
            ref_slope, _ = refs[name]
            err_rel = ((slope - ref_slope) / abs(ref_slope)
                       if ref_slope != 0 else np.nan)
            lines += [
                f"    -- {name} --",
                fmt("analytical celerity", ref_slope, "m s^-1"),
                fmt("fitted speed", slope, "m s^-1"),
                fmt("fitted origin offset (x - x_dam)",
                    intercept - x_dam, "m"),
                fmt("relative speed error", err_rel, ""),
                fmt("residual std about linear fit", resid, "m"),
                "",
            ]
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
