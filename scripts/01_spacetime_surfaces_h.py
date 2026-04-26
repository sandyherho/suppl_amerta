#!/usr/bin/env python
"""Three-dimensional space-time surfaces of water depth h(x, t).

Renders a 2x2 panel of 3D surfaces (one per Riemann problem) sharing a
common colormap range and z-axis, plus a single colorbar on the right.

Inputs : ../data/case_*.nc
Outputs: ../figs/01_spacetime_surfaces_h.{pdf,png,eps}
         ../stats/01_spacetime_surfaces_h.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from netCDF4 import Dataset


# --------------------------------------------------------------- configuration
DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "01_spacetime_surfaces_h"

CASE_FILES = [
    ("stoker",             "case_1_stoker_wet_dam_break.nc"),
    ("ritter",             "case_2_ritter_dry_dam_break.nc"),
    ("double_rarefaction", "case_3_double_rarefaction.nc"),
    ("double_shock",       "case_4_double_shock.nc"),
]

NT_PLOT = 80    # subsampled t resolution for the surface mesh
NX_PLOT = 200   # subsampled x resolution for the surface mesh
ELEV = 25
AZIM = -60
CMAP = "viridis"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "axes.labelpad": 2,
})


# ----------------------------------------------------------------- IO helpers
def load_h_field(path):
    """Read h(time, x), x, t and L from a NetCDF file."""
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as nc:
        return {
            "x": np.asarray(nc.variables["x"][:]),
            "t": np.asarray(nc.variables["time"][:]),
            "h": np.asarray(nc.variables["h"][:]),
            "L": float(nc.L),
        }


def subsample(x, t, h, nx_plot, nt_plot):
    """Uniformly subsample x and t for surface plotting."""
    ix = np.linspace(0, len(x) - 1, min(nx_plot, len(x))).astype(int)
    it = np.linspace(0, len(t) - 1, min(nt_plot, len(t))).astype(int)
    return x[ix], t[it], h[np.ix_(it, ix)]


# ------------------------------------------------------------------- analysis
def shock_ridge_path(x, t, h):
    """Return (t, x) of the steepest-gradient cell at each timestep."""
    grad = np.abs(np.diff(h, axis=1))
    idx = np.argmax(grad, axis=1)
    xc = 0.5 * (x[:-1] + x[1:])
    return t, xc[idx]


def dry_tip_path(x, t, h, h_dry):
    """Return (t, x) of the rightmost cell with h > h_dry at each timestep."""
    pos = np.full(h.shape[0], np.nan)
    for i in range(h.shape[0]):
        wet = np.where(h[i] > h_dry)[0]
        if wet.size:
            pos[i] = x[wet[-1]]
    return t, pos


def linear_speed(t, x, skip=5):
    """Slope of x(t) via least squares, skipping the first `skip` samples."""
    if len(t) <= skip + 2:
        return np.nan
    valid = ~np.isnan(x)
    valid[:skip] = False
    if valid.sum() < 5:
        return np.nan
    return float(np.polyfit(t[valid], x[valid], 1)[0])


# ------------------------------------------------------------------- plotting
def plot_surfaces(cases, vmin, vmax, outpath_stem):
    """Build and save the 2x2 surface figure."""
    fig = plt.figure(figsize=(9.0, 7.2))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.04],
        left=0.04, right=0.93, top=0.97, bottom=0.05,
        hspace=0.18, wspace=0.10,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, (key, _) in zip(tags, positions, CASE_FILES):
        c = cases[key]
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection="3d")
        X, T = np.meshgrid(c["x_sub"], c["t_sub"])
        ax.plot_surface(
            X, T, c["h_sub"],
            cmap=CMAP, vmin=vmin, vmax=vmax,
            rstride=1, cstride=1, linewidth=0,
            antialiased=False, edgecolor="none",
        )
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("t [s]")
        ax.set_zlabel("h [m]")
        ax.set_zlim(vmin, vmax)
        ax.tick_params(pad=0)
        ax.text2D(0.02, 0.96, tag, transform=ax.transAxes,
                  fontsize=11, fontweight="bold", va="top")

    cax = fig.add_subplot(gs[:, 2])
    sm = cm.ScalarMappable(cmap=CMAP,
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("h [m]")

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------- stats
def write_stats(cases, outpath, vmin, vmax):
    """Write per-case extrema and tracked-front diagnostics."""
    lines = [
        "=" * 72,
        "Space-time surfaces of h(x, t)",
        "=" * 72,
        f"Global colormap range : [{vmin:.6f}, {vmax:.6f}] m",
        f"Plot subsampling      : {NT_PLOT} (t) x {NX_PLOT} (x)",
        f"View                  : elev={ELEV}, azim={AZIM}",
        "",
    ]
    for key, _ in CASE_FILES:
        c = cases[key]
        x, t, h = c["x"], c["t"], c["h"]
        idx = np.unravel_index(np.argmax(h), h.shape)
        lines += [
            "-" * 72,
            f"Case: {key}",
            f"  shape (nt, nx)        : ({h.shape[0]}, {h.shape[1]})",
            f"  domain length L       : {c['L']:.4f} m",
            f"  t range               : [{t[0]:.4f}, {t[-1]:.4f}] s",
            f"  x range               : [{x[0]:.4f}, {x[-1]:.4f}] m",
            f"  h_min                 : {h.min():.6e} m",
            f"  h_max                 : {h.max():.6e} m",
            f"  h_mean                : {h.mean():.6e} m",
            f"  argmax h location     : x = {x[idx[1]]:.4f} m, "
            f"t = {t[idx[0]]:.4f} s",
        ]
        if key in ("stoker", "double_shock"):
            tt, xx = shock_ridge_path(x, t, h)
            speed = linear_speed(tt, xx)
            lines.append(f"  shock-ridge speed     : {speed:+.6f} m/s")
        if key == "ritter":
            tt, xx = dry_tip_path(x, t, h, h_dry=0.05)
            speed = linear_speed(tt, xx)
            lines.append(f"  dry-tip speed         : {speed:+.6f} m/s")
        lines.append("")

    outpath.write_text("\n".join(lines))


# ----------------------------------------------------------------------- main
def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    cases = {}
    for key, fname in CASE_FILES:
        d = load_h_field(DATA_DIR / fname)
        x_s, t_s, h_s = subsample(d["x"], d["t"], d["h"], NX_PLOT, NT_PLOT)
        cases[key] = {**d, "x_sub": x_s, "t_sub": t_s, "h_sub": h_s}

    vmin = float(min(c["h"].min() for c in cases.values()))
    vmax = float(max(c["h"].max() for c in cases.values()))

    plot_surfaces(cases, vmin, vmax, FIGS_DIR / OUTPUT_NAME)
    write_stats(cases, STATS_DIR / f"{OUTPUT_NAME}.txt", vmin, vmax)
    print(f"wrote: {FIGS_DIR / OUTPUT_NAME}.{{pdf,png,eps}}")
    print(f"wrote: {STATS_DIR / OUTPUT_NAME}.txt")


if __name__ == "__main__":
    main()
