#!/usr/bin/env python
"""Self-similarity collapse h(x, t) -> H(xi) where xi = (x - x_dam) / t.

Riemann solutions on the real line are self-similar: the depth depends
only on the similarity coordinate xi.  Each panel overlays h(xi) at
several timesteps for one Riemann problem; perfect collapse manifests
as overlapping curves.

Inputs : ../data/case_*.nc
Outputs: ../figs/02_riemann_self_similarity.{pdf,png,eps}
         ../stats/02_riemann_self_similarity.txt
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

N_OVERLAY = 14      # snapshots per panel
T_MIN_FRAC = 0.05   # discard first 5% of t (xi blows up near t=0)
N_BINS = 200        # xi bins for similarity-violation metric

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
    valid = np.where(t > t[-1] * t_min_frac)[0]
    if valid.size == 0:
        return np.array([], dtype=int)
    if valid.size < n:
        return valid
    return valid[np.linspace(0, valid.size - 1, n).astype(int)]


def similarity_violation(x, t, h, x_dam, n_bins, t_min_frac):
    """sigma_t(h) at fixed xi = (x - x_dam) / t."""
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
        x_dam = 0.5 * d["L"]
        idxs = select_overlay_indices(d["t"], N_OVERLAY, T_MIN_FRAC)
        for ii in idxs:
            xi = (d["x"] - x_dam) / d["t"][ii]
            ax.plot(xi, d["h"][ii], color=c["color"], alpha=0.45, lw=0.9)
        ax.set_xlabel(r"$\xi = (x - x_{\mathrm{dam}})/t$  [m s$^{-1}$]")
        ax.set_ylabel(r"$h$  [m]")
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
        if value != value:                          # NaN
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
        "  AMERTA v0.0.3  --  Riemann self-similarity collapse",
        sep,
        "",
        "  Each Riemann solution depends only on xi = (x - x_dam) / t.  This",
        "  report quantifies temporal scatter sigma_t(h) at fixed xi: small",
        "  values indicate near-perfect self-similar collapse.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Snapshots overlaid per panel", N_OVERLAY),
        fmt("Initial fraction of t skipped",
            T_MIN_FRAC * 100.0, "% of t_final"),
        fmt("xi bins for sigma_t metric", N_BINS),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        xi_c, sigma = similarity_violation(
            d["x"], d["t"], d["h"], 0.5 * d["L"], N_BINS, T_MIN_FRAC,
        )
        ok = ~np.isnan(sigma)
        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        if ok.sum() == 0:
            lines += ["    no valid bins.", ""]
            continue
        dxi = xi_c[1] - xi_c[0]
        l1 = float(np.sum(sigma[ok]) * dxi)
        l2 = float(np.sqrt(np.sum(sigma[ok] ** 2) * dxi))
        i_max = int(np.argmax(np.where(ok, sigma, -np.inf)))
        h_mean = float(d["h"].mean())
        quality = 1.0 - float(np.mean(sigma[ok])) / max(h_mean, 1e-12)

        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            fmt("Number of valid xi bins",
                int(ok.sum()), f"/ {N_BINS}"),
            fmt("xi range (min)", float(xi_c[0]), "m s^-1"),
            fmt("xi range (max)", float(xi_c[-1]), "m s^-1"),
            "",
            "    -- self-similarity violation sigma_t(h) --",
            fmt("mean over xi", float(np.mean(sigma[ok])), "m"),
            fmt("max over xi", float(sigma[i_max]), "m"),
            fmt("xi at max sigma", float(xi_c[i_max]), "m s^-1"),
            fmt("L1 of sigma over xi", l1, "m * (m s^-1)"),
            fmt("L2 of sigma over xi", l2, "m * sqrt(m s^-1)"),
            fmt("similarity quality 1 - <sigma>/<h>", quality, ""),
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
