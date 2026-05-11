#!/usr/bin/env python
"""Self-similarity collapse h(x, t) -> h(xi) with analytical overlay.

Inputs : ../data/case_*.nc
Outputs: ../figs/03_self_similarity.{pdf,png,eps}
         ../stats/03_self_similarity.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset


DATA_DIR  = Path("../data")
FIGS_DIR  = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "03_self_similarity"

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

N_OVERLAY      = 12
T_MIN_FRAC     = 0.05
N_BINS_QUALITY = 200

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
            "x":    np.asarray(nc.variables["x"][:]),
            "t":    np.asarray(nc.variables["time"][:]),
            "h":    np.asarray(nc.variables["h"][:]),
            "h_an": np.asarray(nc.variables["h_analytical"][:]),
            "L":    float(nc.L),
        }


def style_axes(ax):
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=6)
    ax.tick_params(direction="out", length=3)
    ax.margins(x=0.02, y=0.06)
    ax.grid(alpha=0.3)


def select_indices(t, n, t_min_frac):
    valid = np.where(t > t[-1] * t_min_frac)[0]
    if valid.size == 0:
        return np.array([], dtype=int)
    if valid.size < n:
        return valid
    return valid[np.linspace(0, valid.size - 1, n).astype(int)]


def similarity_quality(x, t, h, x_dam, n_bins, t_min_frac):
    sel = t > t[-1] * t_min_frac
    if sel.sum() == 0:
        return np.nan, np.nan
    tg, xg = np.meshgrid(t[sel], x, indexing="ij")
    xi = ((xg - x_dam) / tg).ravel()
    hh = h[sel].ravel()
    bins = np.linspace(xi.min(), xi.max(), n_bins + 1)
    sigma = np.full(n_bins, np.nan)
    bin_idx = np.clip(np.searchsorted(bins, xi, side="right") - 1,
                      0, n_bins - 1)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 1:
            sigma[b] = hh[m].std()
    ok = ~np.isnan(sigma)
    if ok.sum() == 0:
        return np.nan, np.nan
    mean_sigma = float(sigma[ok].mean())
    mean_h = float(h.mean())
    return mean_sigma, 1.0 - mean_sigma / max(mean_h, 1e-12)


def plot_figure(data, outpath_stem):
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, left=0.08, right=0.97, top=0.97, bottom=0.13,
        hspace=0.30, wspace=0.24,
    )
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for tag, pos, c in zip(tags, positions, CASES):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        d = data[c["key"]]
        x_dam = 0.5 * d["L"]
        idxs = select_indices(d["t"], N_OVERLAY, T_MIN_FRAC)
        for ii in idxs:
            xi = (d["x"] - x_dam) / d["t"][ii]
            ax.plot(xi, d["h"][ii], color=c["color"], alpha=0.45, lw=0.8)
        xi_an = (d["x"] - x_dam) / d["t"][-1]
        order = np.argsort(xi_an)
        ax.plot(xi_an[order], d["h_an"][-1][order],
                color="k", lw=1.4, ls="--", alpha=0.9)
        ax.set_xlabel(r"$\xi = (x - x_{\mathrm{dam}})/t$  [m s$^{-1}$]")
        ax.set_ylabel(r"$h$  [m]")
        ax.set_title(tag, loc="left", fontweight="bold",
                     fontsize=11, pad=4)
        style_axes(ax)

    proxies = [Line2D([0], [0], color=c["color"], lw=2.0, label=c["label"])
               for c in CASES]
    proxies.append(Line2D([0], [0], color="k", lw=1.4, ls="--",
                          label="analytical"))
    fig.legend(handles=proxies, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.015), frameon=False,
               fontsize=9.5, handlelength=2.4, columnspacing=2.0)

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
        "  AMERTA  --  Riemann self-similarity collapse",
        sep,
        "",
        "  For each case h(x, t) is reparameterized as h(xi) with",
        "      xi = (x - x_dam) / t.",
        "  Temporal scatter sigma_t(h) at fixed xi is binned and",
        "  averaged over xi; quality = 1 - <sigma>/<h>.",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Snapshots overlaid per panel", N_OVERLAY),
        fmt("Initial fraction of t skipped",
            T_MIN_FRAC * 100.0, "% of t_final"),
        fmt("xi bins for sigma metric", N_BINS_QUALITY),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        sig_mean, q = similarity_quality(
            d["x"], d["t"], d["h"], 0.5 * d["L"],
            N_BINS_QUALITY, T_MIN_FRAC,
        )
        lines += [
            sub,
            f"  CASE {i} / 4  --  {c['label']}",
            sub,
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Grid points nx", int(d["x"].size)),
            fmt("Snapshots stored nt", int(d["t"].size)),
            fmt("<sigma_t(h)>_xi", float(sig_mean), "m"),
            fmt("<h>", float(d["h"].mean()), "m"),
            fmt("similarity quality 1 - <sigma>/<h>", float(q), ""),
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
