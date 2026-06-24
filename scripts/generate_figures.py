"""Export all report figures as PNG files for the markdown report.

Usage:
    python scripts/generate_figures.py --out docs/figures
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROTEINS = ["3u7t_A", "4p3a_B", "1b2s_F", "2y4x_B", "1z0b_A", "6ovk_R"]
PROTEIN_LABELS = ["3u7t_A\n(46 aa)", "4p3a_B\n(79 aa)", "1b2s_F\n(90 aa)",
                  "2y4x_B\n(93 aa)", "1z0b_A\n(207 aa)", "6ovk_R\n(219 aa)"]
TEMPS = [300, 375, 450]
TEMP_COLORS = {"300": "#2166ac", "375": "#4dac26", "450": "#d6604d"}

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_v4_metric(protein, temp_K, key_path):
    d = load_json(f"validation_v4_{protein}_T{temp_K}.json")
    if d is None:
        return float("nan")
    try:
        obj = d
        for k in key_path:
            obj = obj[k]
        return float(obj) if obj is not None else float("nan")
    except (KeyError, TypeError):
        return float("nan")


def get_v4_best(protein, metric_path):
    vals = {T: get_v4_metric(protein, T, metric_path) for T in TEMPS}
    if "rmsf_corr" in str(metric_path):
        best_T = max(vals, key=lambda t: vals[t] if not math.isnan(vals[t]) else -1)
    else:
        best_T = min(vals, key=lambda t: vals[t] if not math.isnan(vals[t]) else 1e9)
    return vals[best_T], best_T


def get_v2_metric(protein, key):
    d = load_json(f"eval_{protein}.json")
    if d is None:
        return float("nan")
    return float(d.get("model", {}).get(key, float("nan")))


def get_v3_metric(protein, key):
    d = load_json("validation_v3_lam0.json")
    if d is None:
        return float("nan")
    try:
        return float(d["proteins"][protein]["structural"][key])
    except (KeyError, TypeError):
        return float("nan")


def get_longlags_metric(protein, key_path):
    d = load_json("validation_v4_longlags_T300.json")
    if d is None:
        return float("nan")
    try:
        obj = d["proteins"][protein]
        for k in key_path:
            obj = obj[k]
        return float(obj)
    except (KeyError, TypeError):
        return float("nan")


def save(fig, out_dir, name, dpi=150):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
    return path


# ── Fig 1: Lag strategy ───────────────────────────────────────────────────────

def fig_lag_strategy(out_dir):
    with plt.rc_context({**STYLE, "axes.grid": False, "axes.spines.left": False,
                         "axes.spines.bottom": True}):
        fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.set_xlim(80, 60000)
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["v3 (failed)", "v4 attempt 1\n(lags ≥ 2k ps)",
                         "v4 final\n(lags ≥ 100 ps)", "Inference τ"], fontsize=9)
    ax.set_xlabel("Lag / picoseconds (log scale)", fontsize=10)
    ax.set_title("Evolution of Training Lag Strategy", fontsize=12, fontweight="bold")

    for lag in [5000, 10000, 20000]:
        ax.plot(lag, 0, "o", color="#d73027", markersize=10, zorder=3)
    ax.annotate("OOD gap →", xy=(2000, 0), xytext=(2000, 0.55), fontsize=7.5,
                color="#d73027", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2))
    for lag in [2000, 5000, 10000, 20000, 30000, 50000]:
        ax.plot(lag, 1, "s", color="#fc8d59", markersize=9, zorder=3)
    for lag in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        ax.plot(lag, 2, "D", color="#1a9641", markersize=8, zorder=3)

    ax.axvline(2000, color="#4575b4", linewidth=1.8, linestyle="--", alpha=0.7)
    ax.plot(2000, 3, "^", color="#4575b4", markersize=11, zorder=3)
    ax.text(2000, 3.25, "τ = 2000 ps", ha="center", fontsize=8, color="#4575b4",
            fontweight="bold")
    ax.axvline(100, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(105, 3.25, "min frame\n(100 ps)", ha="left", fontsize=7.5, color="gray")

    legend_elements = [
        mpatches.Patch(color="#d73027", label="v3: lags [5k, 10k, 20k ps] — τ=2000 below range → NaN"),
        mpatches.Patch(color="#fc8d59", label="v4 attempt 1: add 2k ps floor"),
        mpatches.Patch(color="#1a9641", label="v4 final: full range 100 ps → 50k ps"),
        mpatches.Patch(color="#4575b4", label="Inference τ = 2000 ps"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    return save(fig, out_dir, "fig1_lag_strategy.png")


# ── Fig 2: Model progression ──────────────────────────────────────────────────

def fig_progression(out_dir):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))

    v2 = [get_v2_metric(p, "rmsf_corr") for p in PROTEINS]
    v3 = [get_v3_metric(p, "rmsf_corr") for p in PROTEINS]
    vLL = [get_longlags_metric(p, ["structural", "rmsf_corr"]) for p in PROTEINS]
    v4b = [get_v4_best(p, ["proteins", p, "structural", "rmsf_corr"])[0] for p in PROTEINS]

    x = np.arange(len(PROTEINS))
    w = 0.19
    ax.bar(x - 1.5*w, v2,  width=w, label="v2 (baseline)", color="#aec7e8", edgecolor="white")
    ax.bar(x - 0.5*w, v3,  width=w, label="v3 (ATLAS fine-tune)", color="#ffbb78", edgecolor="white")
    ax.bar(x + 0.5*w, vLL, width=w, label="v4 long-lag universal", color="#98df8a", edgecolor="white")
    ax.bar(x + 1.5*w, v4b, width=w, label="v4 per-protein (best T)", color="#1f77b4", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(PROTEIN_LABELS, fontsize=9)
    ax.set_ylabel("RMSF correlation (Pearson r)", fontsize=10)
    ax.set_title("RMSF Profile Correlation Across Model Versions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    for xi, val in zip(x + 1.5*w, v4b):
        if not math.isnan(val):
            ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#1f77b4")
    fig.tight_layout()
    return save(fig, out_dir, "fig2_progression.png")


# ── Fig 3: Best-temperature summary ───────────────────────────────────────────

def fig_best_temp_summary(out_dir):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1, ax2 = axes
    best_rmsf, best_fes, best_T_rmsf, best_T_fes = [], [], [], []
    for p in PROTEINS:
        r, T_r = get_v4_best(p, ["proteins", p, "structural", "rmsf_corr"])
        f, T_f = get_v4_best(p, ["proteins", p, "thermodynamic", "fes_js"])
        best_rmsf.append(r); best_fes.append(f)
        best_T_rmsf.append(T_r); best_T_fes.append(T_f)

    x = np.arange(len(PROTEINS))
    ax1.bar(x, best_rmsf, color=[TEMP_COLORS[str(T)] for T in best_T_rmsf],
            edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5,
                         rotation=15, ha="right")
    ax1.set_ylabel("RMSF Correlation (Pearson r)", fontsize=9.5)
    ax1.set_title("Best RMSF Correlation", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for xi, (val, T) in enumerate(zip(best_rmsf, best_T_rmsf)):
        ax1.text(xi, val + 0.01, f"{val:.3f}\n@{T}K", ha="center", va="bottom",
                 fontsize=7.5)

    ax2.bar(x, best_fes, color=[TEMP_COLORS[str(T)] for T in best_T_fes],
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5,
                         rotation=15, ha="right")
    ax2.set_ylabel("FES JS Divergence (lower = better)", fontsize=9.5)
    ax2.set_title("Best FES JS Divergence", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for xi, (val, T) in enumerate(zip(best_fes, best_T_fes)):
        ax2.text(xi, val + 0.01, f"{val:.3f}\n@{T}K", ha="center", va="bottom",
                 fontsize=7.5)

    legend_elements = [mpatches.Patch(color=TEMP_COLORS["300"], label="300 K"),
                       mpatches.Patch(color=TEMP_COLORS["375"], label="375 K"),
                       mpatches.Patch(color=TEMP_COLORS["450"], label="450 K")]
    fig.legend(handles=legend_elements, title="Best temperature", fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, out_dir, "fig3_best_temp.png")


# ── Fig 4: Temperature heatmap ────────────────────────────────────────────────

def fig_temp_heatmap(out_dir):
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    metrics = [
        ("RMSF Correlation", ["proteins", "{p}", "structural", "rmsf_corr"], True),
        ("FES JS Divergence", ["proteins", "{p}", "thermodynamic", "fes_js"], False),
        ("Relaxation Ratio", ["proteins", "{p}", "kinetic", "relax_ratio"], False),
    ]
    for ax, (title, path_tmpl, higher_better) in zip(axes, metrics):
        data = np.zeros((len(PROTEINS), 3))
        for i, p in enumerate(PROTEINS):
            for j, T in enumerate(TEMPS):
                path = [k.replace("{p}", p) for k in path_tmpl]
                data[i, j] = get_v4_metric(p, T, path)

        if higher_better:
            cmap, vmin, vmax = "RdYlGn", 0, 1
        elif "js" in title.lower():
            cmap, vmin, vmax = "RdYlGn_r", 0, 1
        else:
            cmap, vmin, vmax = "RdYlGn_r", 0, 15

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["300 K", "375 K", "450 K"], fontsize=9)
        ax.set_yticks(range(len(PROTEINS)))
        ax.set_yticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        for i in range(len(PROTEINS)):
            for j in range(3):
                v = data[i, j]
                txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
                color = ("white" if (higher_better and v < 0.4) or
                         (not higher_better and v > 0.6 * vmax) else "black")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("v4 Per-Protein Validation: Temperature Sweep", fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return save(fig, out_dir, "fig4_heatmap.png")


# ── Fig 5: Kinetics ───────────────────────────────────────────────────────────

def fig_kinetics(out_dir):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(PROTEINS))
    w = 0.25
    for j, T in enumerate(TEMPS):
        vals = [get_v4_metric(p, T, ["proteins", p, "kinetic", "relax_ratio"])
                for p in PROTEINS]
        ax.bar(x + (j - 1)*w, vals, width=w, label=f"{T} K",
               color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)

    ax.axhline(1.0, color="black", linewidth=1.4, linestyle="--", label="Ideal (ratio = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=9)
    ax.set_ylabel("Relaxation time ratio  τ_model / τ_MD", fontsize=10)
    ax.set_title("Kinetics: Model vs MD Relaxation Time Ratio", fontsize=12,
                 fontweight="bold")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return save(fig, out_dir, "fig5_kinetics.png")


# ── Fig 6: Structural geometry ────────────────────────────────────────────────

def fig_structural(out_dir):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1, ax2 = axes
    for j, T in enumerate(TEMPS):
        x = np.arange(len(PROTEINS))
        w = 0.25
        bonds   = [get_v4_metric(p, T, ["proteins", p, "structural", "ca_bond_mean"])
                   for p in PROTEINS]
        clashes = [get_v4_metric(p, T, ["proteins", p, "structural", "clash_count"])
                   for p in PROTEINS]
        ax1.bar(x + (j-1)*w, bonds,   width=w, label=f"{T} K",
                color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)
        ax2.bar(x + (j-1)*w, clashes, width=w, label=f"{T} K",
                color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)

    ax1.axhline(3.81, color="black", linewidth=1.3, linestyle="--", label="MD ideal (3.81 Å)")
    ax1.set_xticks(np.arange(len(PROTEINS)))
    ax1.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5,
                         rotation=15, ha="right")
    ax1.set_ylabel("Mean Cα–Cα bond length (Å)", fontsize=9)
    ax1.set_title("Cα Bond Length", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.set_xticks(np.arange(len(PROTEINS)))
    ax2.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5,
                         rotation=15, ha="right")
    ax2.set_ylabel("Mean clashes per frame", fontsize=9)
    ax2.set_title("Cα Clash Count", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("v4 Per-Protein Structural Geometry", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, "fig6_structural.png")


# ── Fig 7: MD acceleration estimate ──────────────────────────────────────────

def fig_acceleration(out_dir):
    # Throughput: ~540 µs/day (pure rollout estimate from pipeline timing)
    # Classical MD: 100–300 ns/day for explicit-solvent ~100 aa on A100
    tp_lo, tp_hi = 1800, 5400   # raw throughput speedup range

    best_T = [375, 375, 300, 375, 300, 375]
    relax_ratios = []
    for p, T in zip(PROTEINS, best_T):
        rr = get_v4_metric(p, T, ["proteins", p, "kinetic", "relax_ratio"])
        relax_ratios.append(rr)

    eff_lo = [tp_lo / rr for rr in relax_ratios]
    eff_hi = [tp_hi / rr for rr in relax_ratios]
    eff_mid = [(lo + hi) / 2 for lo, hi in zip(eff_lo, eff_hi)]
    eff_err = [(hi - lo) / 2 for lo, hi in zip(eff_lo, eff_hi)]

    n_res = [46, 79, 90, 93, 207, 219]
    # colour by size: small=blue, large=orange
    colours = ["#2166ac" if n < 150 else "#d6604d" for n in n_res]

    with plt.rc_context({**STYLE, "axes.grid.axis": "x"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1, ax2 = axes
    x = np.arange(len(PROTEINS))

    # Left: effective speedup (log scale)
    bars = ax1.barh(x, eff_mid, xerr=eff_err, color=colours, edgecolor="white",
                    height=0.55, capsize=4, error_kw={"elinewidth": 1.2})
    ax1.set_xscale("log")
    ax1.set_yticks(x)
    ax1.set_yticklabels([f"{p.replace('_',' ')}\n({n} aa)"
                         for p, n in zip(PROTEINS, n_res)], fontsize=9)
    ax1.set_xlabel("Effective speedup vs classical MD (×)", fontsize=9.5)
    ax1.set_title("Estimated MD Acceleration\n(throughput × kinetic factor)",
                  fontsize=10, fontweight="bold")
    ax1.axvline(1000, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax1.axvline(10000, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    for xi, (mid, lo, hi) in enumerate(zip(eff_mid, eff_lo, eff_hi)):
        ax1.text(hi * 1.15, xi, f"{lo:.0f}–{hi:.0f}×",
                 va="center", fontsize=7.5, color="#333")

    # Right: decomposition — throughput vs kinetic factor
    w = 0.35
    ax2.bar(x - w/2, [tp_lo]*len(PROTEINS), width=w, label="Throughput (lower bound)",
            color="#98df8a", edgecolor="white", alpha=0.9)
    ax2.bar(x + w/2, [1/rr for rr in relax_ratios], width=w,
            label="Kinetic factor (1 / relax_ratio)", color="#aec7e8", edgecolor="white", alpha=0.9)
    ax2.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5,
                         rotation=15, ha="right")
    ax2.set_ylabel("Factor", fontsize=9.5)
    ax2.set_title("Throughput vs Kinetic Contribution",
                  fontsize=10, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper right")

    legend_elements = [mpatches.Patch(color="#2166ac", label="Small protein (< 150 aa)"),
                       mpatches.Patch(color="#d6604d", label="Large protein (≥ 150 aa)")]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    return save(fig, out_dir, "fig7_acceleration.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Generating figures...")
    fig_lag_strategy(args.out)
    fig_progression(args.out)
    fig_best_temp_summary(args.out)
    fig_temp_heatmap(args.out)
    fig_kinetics(args.out)
    fig_structural(args.out)
    fig_acceleration(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
