#!/usr/bin/env python
"""Entropy and statistical-complexity diagnostics of the depth field.

Treating the normalized depth p_i(t) = h_i(t) / sum_j h_j(t) as a
discrete probability distribution over grid cells, this script
computes four information-theoretic measures versus time:

  (a) normalized spatial Shannon entropy
          H_s(t) = -sum_i p_i log p_i  /  log N,
  (b) Bandt-Pompe normalized permutation entropy of the spatial
      profile h(:, t),
          PE_m(t) = -sum_pi pi log pi  /  log m!,
      using embedding dimension m and delay tau,
  (c) Lopez-Ruiz-Mancini-Calbet disequilibrium
          D(t) = sum_i (p_i - 1/N)^2,
  (d) LMC statistical complexity
          C(t) = H_s(t) * D(t).

H_s captures spread, PE captures ordinal pattern complexity, D
captures distance from uniform, and C peaks at structured intermediate
states between maximum order and maximum disorder.

Inputs : ../data/case_*.nc
Outputs: ../figs/07_entropy_complexity.{pdf,png,eps}
         ../stats/07_entropy_complexity.txt
"""
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset


# --------------------------------------------------------------- configuration
DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")
OUTPUT_NAME = "07_entropy_complexity"

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

PE_M = 3              # Bandt-Pompe embedding dimension
PE_TAU = 1            # Bandt-Pompe delay
EPS_PROB = 1.0e-300   # numerical floor for probabilities

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


# ---------------------------------------------------------- entropy measures
def normalize_to_probability(values, eps=EPS_PROB):
    """Return p such that sum(p) == 1 and p >= 0; None if unnormalizable."""
    v = np.maximum(np.asarray(values, dtype=float), 0.0)
    s = v.sum()
    if not np.isfinite(s) or s <= eps:
        return None
    return v / s


def shannon_entropy_normalized(values, eps=EPS_PROB):
    p = normalize_to_probability(values, eps)
    if p is None:
        return np.nan
    pp = p[p > eps]
    return float(-np.sum(pp * np.log(pp)) / np.log(len(p)))


def disequilibrium(values, eps=EPS_PROB):
    p = normalize_to_probability(values, eps)
    if p is None:
        return np.nan
    return float(np.sum((p - 1.0 / len(p)) ** 2))


def lmc_complexity(values, eps=EPS_PROB):
    p = normalize_to_probability(values, eps)
    if p is None:
        return np.nan
    pp = p[p > eps]
    h_norm = float(-np.sum(pp * np.log(pp)) / np.log(len(p)))
    d = float(np.sum((p - 1.0 / len(p)) ** 2))
    return h_norm * d


def permutation_entropy_normalized(series, m=PE_M, tau=PE_TAU):
    """Bandt-Pompe permutation entropy normalized to [0, 1]."""
    series = np.asarray(series)
    n = len(series)
    if n < m * tau + 1 or m < 2:
        return np.nan
    counts = {}
    for i in range(n - (m - 1) * tau):
        window = series[i:i + m * tau:tau]
        order = tuple(int(k) for k in np.argsort(window, kind="stable"))
        counts[order] = counts.get(order, 0) + 1
    c = np.array(list(counts.values()), dtype=float)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)) / np.log(factorial(m)))


# --------------------------------------------------- per-case time-series fns
def entropy_series(h):
    """H_s(t), D(t), C(t), PE_m(t) computed row-wise."""
    nt = h.shape[0]
    Hs = np.array([shannon_entropy_normalized(h[i]) for i in range(nt)])
    D = np.array([disequilibrium(h[i]) for i in range(nt)])
    C = np.array([lmc_complexity(h[i]) for i in range(nt)])
    PE = np.array([permutation_entropy_normalized(h[i]) for i in range(nt)])
    return Hs, PE, D, C


# ------------------------------------------------------------------- plotting
def plot_figure(data, series, outpath_stem):
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
    ylabels = [
        r"normalized Shannon entropy  $H_s$",
        r"permutation entropy  $\mathrm{PE}_{" + str(PE_M) + r"}$",
        r"disequilibrium  $D$",
        r"LMC complexity  $C = H_s \cdot D$",
    ]

    for k, (ax, tag, ylab) in enumerate(zip(axes, tags, ylabels)):
        for c in CASES:
            d = data[c["key"]]
            ax.plot(d["t"], series[c["key"]][k], color=c["color"])
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(ylab)
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


def write_stats(data, series, outpath):
    sep = "=" * 80
    sub = "-" * 80
    lines = [
        sep,
        "  AMERTA v0.0.3  --  Entropy and statistical-complexity diagnostics",
        sep,
        "",
        "  Treating p_i(t) = h_i(t) / sum_j h_j(t) as a probability mass",
        "  function on grid cells, this report tracks four information-",
        "  theoretic measures of the depth field:",
        "      H_s  = normalized spatial Shannon entropy in [0, 1]",
        "      PE_m = normalized Bandt-Pompe permutation entropy in [0, 1]",
        "      D    = disequilibrium  sum_i (p_i - 1/N)^2",
        "      C    = LMC statistical complexity  H_s * D",
        "",
        sub,
        "  CONFIGURATION",
        sub,
        fmt("Permutation entropy embedding m", PE_M),
        fmt("Permutation entropy delay tau", PE_TAU),
        "",
    ]
    for i, c in enumerate(CASES, 1):
        d = data[c["key"]]
        Hs, PE, D, C = series[c["key"]]
        # finite-only stats
        Hs_f = Hs[np.isfinite(Hs)]
        PE_f = PE[np.isfinite(PE)]
        D_f = D[np.isfinite(D)]
        C_f = C[np.isfinite(C)]

        lines += [sub, f"  CASE {i} / 4  --  {c['label']}", sub]
        lines += [
            fmt("Domain length L", float(d["L"]), "m"),
            fmt("Final time t_f", float(d["t"][-1]), "s"),
            fmt("Grid points nx", int(d["h"].shape[1])),
            "",
            "    -- normalized Shannon entropy H_s(t) --",
            fmt("H_s(0)", float(Hs[0]) if Hs.size else np.nan, ""),
            fmt("H_s(t_f)",
                float(Hs[-1]) if Hs.size else np.nan, ""),
            fmt("min over t",
                float(Hs_f.min()) if Hs_f.size else np.nan, ""),
            fmt("max over t",
                float(Hs_f.max()) if Hs_f.size else np.nan, ""),
            "",
            f"    -- permutation entropy PE_{PE_M}(t) --",
            fmt(f"PE_{PE_M}(0)",
                float(PE[0]) if PE.size else np.nan, ""),
            fmt(f"PE_{PE_M}(t_f)",
                float(PE[-1]) if PE.size else np.nan, ""),
            fmt("min over t",
                float(PE_f.min()) if PE_f.size else np.nan, ""),
            fmt("max over t",
                float(PE_f.max()) if PE_f.size else np.nan, ""),
            "",
            "    -- disequilibrium D(t) --",
            fmt("D(0)", float(D[0]) if D.size else np.nan, ""),
            fmt("D(t_f)", float(D[-1]) if D.size else np.nan, ""),
            fmt("min over t",
                float(D_f.min()) if D_f.size else np.nan, ""),
            fmt("max over t",
                float(D_f.max()) if D_f.size else np.nan, ""),
            "",
            "    -- LMC statistical complexity C(t) = H_s * D --",
            fmt("C(0)", float(C[0]) if C.size else np.nan, ""),
            fmt("C(t_f)", float(C[-1]) if C.size else np.nan, ""),
            fmt("min over t",
                float(C_f.min()) if C_f.size else np.nan, ""),
            fmt("max over t",
                float(C_f.max()) if C_f.size else np.nan, ""),
            fmt("argmax_t C  [s]",
                float(d["t"][np.nanargmax(C)])
                if np.any(np.isfinite(C)) else np.nan, "s"),
            "",
        ]
    outpath.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    data = {c["key"]: load_case(DATA_DIR / c["file"]) for c in CASES}
    series = {c["key"]: entropy_series(data[c["key"]]["h"]) for c in CASES}
    plot_figure(data, series, FIGS_DIR / OUTPUT_NAME)
    write_stats(data, series, STATS_DIR / f"{OUTPUT_NAME}.txt")
    print(f"wrote: {FIGS_DIR / OUTPUT_NAME}.{{pdf,png,eps}}")
    print(f"wrote: {STATS_DIR / OUTPUT_NAME}.txt")


if __name__ == "__main__":
    main()
