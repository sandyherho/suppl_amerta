#!/usr/bin/env python
"""Self-similarity collapse h(x, t) -> H(xi) where xi = (x - x_dam) / t.

Riemann solutions on the real line are self-similar: the depth depends
only on the similarity coordinate xi.  This script overlays h(xi) at
several timesteps for each case and quantifies the temporal scatter
sigma_t(h) at fixed xi.

Inputs : ../data/case_*.nc
Outputs: ../figs/02_riemann_self_similarity.{pdf,png,eps}
         ../stats/02_riemann_self_similarity.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


# --------------------------------------------------------------- configuration
DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "02_riemann_self_similarity"

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

N_OVERLAY = 12      # number of t snapshots overlaid per panel
T_MIN_FRAC = 0.05   # skip first 5% of t (xi blows up near t=0)
N_BINS = 200        # number of xi bins for the violation panel

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
            "x": np.asarray(nc.variables["x"][:]),
            "t": np.asarray(nc.variables["time"][:]),
            "h": np.asarray(nc.variables["h"][:]),
            "L": float(nc.L),
        }


# ------------------------------------------------------------------- analysis
def select_overlay_indices(t, n, t_min_frac):
    """Return n indices of t (after skipping the first t_min_frac fraction)."""
    valid = np.where(t > t[-1] * t_min_frac)[0]
    if valid.size == 0:
        return np.array([], dtype=int)
    if valid.size < n:
        return valid
    return valid[np.linspace(0, valid.size - 1, n).astype(int)]


def similarity_violation(x, t, h, x_dam, n_bins, t_min_frac):
    """Compute sigma_t(h) at fixed xi.

    Bin all (x, t) samples by xi = (x - x_dam) / t and report the standard
    deviation of h within each bin.  Bins with fewer than 2 samples are NaN.
    """
    sel = t > t[-1] * t_min_frac
    if sel.sum() == 0:
        return np.array([]), np.array([])
    tg, xg = np.meshgrid(t[sel], x, indexing="ij")
    xi = ((xg - x_dam) / tg).ravel()
    hh = h[sel].ravel()

    bins = np.linspace(xi.min(), xi.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sigma = np.full(n_bins, np.nan)
    bin_idx = np.clip(np.searchsorted(bins, xi, side="right") - 1,
                      0, n_bins - 1)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 1:
            sigma[b] = hh[m].std()
    return centers, sigma


# ------------------------------------------------------------------- plotting
def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(11.0, 6.5))
    gs = fig.add_gridspec(
        2, 3, left=0.07, right=0.985, top=0.97, bottom=0.10,
        hspace=0.32, wspace=0.30,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[:, 2]),
    ]
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for ax, tag, c in zip(axes[:4], tags[:4], CASES):
        d = data[c["key"]]
        x_dam = 0.5 * d["L"]
        idxs = select_overlay_indices(d["t"], N_OVERLAY, T_MIN_FRAC)
        for ii in idxs:
            xi = (d["x"] - x_dam) / d["t"][ii]
            ax.plot(xi, d["h"][ii], color=c["color"], alpha=0.45, lw=0.9)
        ax.set_xlabel(r"$\xi = (x - x_{\mathrm{dam}})/t$ [m s$^{-1}$]")
        ax.set_ylabel(r"$h$ [m]")
        ax.text(0.03, 0.95, tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        ax.grid(alpha=0.3)

    ax = axes[4]
    for c in CASES:
        d = data[c["key"]]
        xi_c, sigma = similarity_violation(
            d["x"], d["t"], d["h"], 0.5 * d["L"], N_BINS, T_MIN_FRAC,
        )
        ax.plot(xi_c, sigma, color=c["color"], lw=1.1, label=c["label"])
    ax.set_xlabel(r"$\xi$ [m s$^{-1}$]")
    ax.set_ylabel(r"$\sigma_t(h)$ [m]")
    ax.text(0.03, 0.95, tags[4], transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(outpath_stem.with_suffix(f".{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------- stats
def write_stats(data, outpath):
    lines = [
        "=" * 72,
        "Riemann self-similarity diagnostics",
        "=" * 72,
        f"Skipped first {T_MIN_FRAC * 100:.1f}% of timesteps",
        f"xi bins for sigma(xi) : {N_BINS}",
        f"Overlay snapshots/panel: {N_OVERLAY}",
        "",
    ]
    for c in CASES:
        d = data[c["key"]]
        xi_c, sigma = similarity_violation(
            d["x"], d["t"], d["h"], 0.5 * d["L"], N_BINS, T_MIN_FRAC,
        )
        ok = ~np.isnan(sigma)
        if ok.sum() == 0:
            lines += [f"Case {c['key']}: no valid bins.", ""]
            continue
        dxi = xi_c[1] - xi_c[0]
        l1 = float(np.sum(sigma[ok]) * dxi)
        l2 = float(np.sqrt(np.sum(sigma[ok] ** 2) * dxi))
        sigma_mean = float(np.mean(sigma[ok]))
        i_max = int(np.argmax(np.where(ok, sigma, -np.inf)))
        lines += [
            "-" * 72,
            f"Case: {c['key']}  ({c['label']})",
            f"  xi range                 : [{xi_c[0]:+.4f}, "
            f"{xi_c[-1]:+.4f}] m/s",
            f"  number of valid bins     : {int(ok.sum())} / {N_BINS}",
            f"  mean sigma(xi)           : {sigma_mean:.6e} m",
            f"  max sigma(xi)            : {sigma[i_max]:.6e} m",
            f"  xi at max sigma          : {xi_c[i_max]:+.4f} m/s",
            f"  L1 of sigma(xi)          : {l1:.6e} m * (m/s)",
            f"  L2 of sigma(xi)          : {l2:.6e} m * sqrt(m/s)",
            "",
        ]
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
