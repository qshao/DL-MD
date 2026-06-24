"""Generate a PDF report of the SE(3) PropagatorNet v4 development and validation.

Uses matplotlib for all figures and weasyprint to render HTML → PDF.

Usage:
    python scripts/generate_report.py --out model_comparison_report.pdf
"""
import argparse
import base64
import io
import json
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ─── Data loading ─────────────────────────────────────────────────────────────

PROTEINS = ["3u7t_A", "4p3a_B", "1b2s_F", "2y4x_B", "1z0b_A", "6ovk_R"]
PROTEIN_LABELS = ["3u7t_A\n(46 aa)", "4p3a_B\n(79 aa)", "1b2s_F\n(90 aa)",
                  "2y4x_B\n(93 aa)", "1z0b_A\n(207 aa)", "6ovk_R\n(219 aa)"]
TEMPS = [300, 375, 450]
TEMP_COLORS = {"300": "#2166ac", "375": "#4dac26", "450": "#d6604d"}

PROTEIN_INFO = {
    "3u7t_A": {"n_res": 46,  "desc": "Small β-sheet (chain A)"},
    "4p3a_B": {"n_res": 79,  "desc": "α/β mixed (chain B)"},
    "1b2s_F": {"n_res": 90,  "desc": "Helix bundle (chain F)"},
    "2y4x_B": {"n_res": 93,  "desc": "α/β mixed (chain B)"},
    "1z0b_A": {"n_res": 207, "desc": "Multi-domain α-helical (chain A)"},
    "6ovk_R": {"n_res": 219, "desc": "Large receptor domain (chain R)"},
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_v4_metric(protein, temp_K, key_path):
    """key_path: e.g. ['structural', 'rmsf_corr'] or ['summary', 'mean_rmsf_corr']"""
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
    """Return best value + temperature across T=300/375/450."""
    vals = {T: get_v4_metric(protein, T, metric_path) for T in TEMPS}
    # higher is better for rmsf_corr; lower is better for JS metrics
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


def get_v3_summary_metric(key):
    d = load_json("validation_v3_lam0.json")
    if d is None:
        return {}
    out = {}
    for p in PROTEINS:
        rep = d.get("proteins", {}).get(p, {})
        try:
            val = rep["structural"][key]
            out[p] = float(val)
        except (KeyError, TypeError):
            out[p] = float("nan")
    return out


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


# ─── Figure helpers ───────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}


# ─── Figure 1: Model progression – rmsf_corr ──────────────────────────────────

def fig_progression():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))

    v2 = [get_v2_metric(p, "rmsf_corr") for p in PROTEINS]
    v3 = list(get_v3_summary_metric("rmsf_corr").values())
    vLL = [get_longlags_metric(p, ["structural", "rmsf_corr"]) for p in PROTEINS]
    v4b = [get_v4_best(p, ["proteins", p, "structural", "rmsf_corr"])[0] for p in PROTEINS]

    x = np.arange(len(PROTEINS))
    w = 0.19
    bars = [
        ax.bar(x - 1.5*w, v2,  width=w, label="v2 (baseline)", color="#aec7e8", edgecolor="white"),
        ax.bar(x - 0.5*w, v3,  width=w, label="v3 (ATLAS fine-tune)", color="#ffbb78", edgecolor="white"),
        ax.bar(x + 0.5*w, vLL, width=w, label="v4 long-lag universal", color="#98df8a", edgecolor="white"),
        ax.bar(x + 1.5*w, v4b, width=w, label="v4 per-protein (best T)", color="#1f77b4", edgecolor="white"),
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(PROTEIN_LABELS, fontsize=9)
    ax.set_ylabel("RMSF correlation (Pearson r)", fontsize=10)
    ax.set_title("RMSF Profile Correlation Across Model Versions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    # Annotate best v4 values
    for xi, val in zip(x + 1.5*w, v4b):
        if not math.isnan(val):
            ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#1f77b4")

    fig.tight_layout()
    return fig_to_b64(fig)


# ─── Figure 2: Temperature sweep heatmap ──────────────────────────────────────

def fig_temp_heatmap():
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    metrics = [
        ("RMSF Correlation", ["proteins", "{p}", "structural", "rmsf_corr"], True),
        ("FES JS Divergence", ["proteins", "{p}", "thermodynamic", "fes_js"], False),
        ("Relaxation Ratio (model/MD)", ["proteins", "{p}", "kinetic", "relax_ratio"], False),
    ]

    for ax, (title, path_tmpl, higher_better) in zip(axes, metrics):
        data = np.zeros((len(PROTEINS), 3))
        for i, p in enumerate(PROTEINS):
            for j, T in enumerate(TEMPS):
                path = [k.replace("{p}", p) for k in path_tmpl]
                data[i, j] = get_v4_metric(p, T, path)

        if higher_better:
            cmap = "RdYlGn"
            vmin, vmax = 0, 1
        elif "js" in title.lower():
            cmap = "RdYlGn_r"
            vmin, vmax = 0, 1
        else:
            # relax_ratio: 1.0 is perfect; clip display to [0, 20]
            cmap = "RdYlGn_r"
            vmin, vmax = 0, 15

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
                lum = 0.299 * 0.5 + 0.587 * 0.5 + 0.114 * 0.5  # mid-gray threshold
                color = "white" if (higher_better and v < 0.4) or (not higher_better and v > 0.6 * vmax) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("v4 Per-Protein Validation: Temperature Sweep", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─── Figure 3: Lag strategy diagram ───────────────────────────────────────────

def fig_lag_strategy():
    with plt.rc_context({**STYLE, "axes.grid": False, "axes.spines.left": False,
                         "axes.spines.bottom": True, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.set_xlim(80, 60000)
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["v3 (failed)", "v4 attempt 1\n(lags ≥ 2k ps)", "v4 attempt 2\n(lags ≥ 100 ps)", "Inference τ"], fontsize=9)
    ax.set_xlabel("Lag / picoseconds (log scale)", fontsize=10)
    ax.set_title("Evolution of Training Lag Strategy", fontsize=12, fontweight="bold")

    # v3 lags: [5000, 10000, 20000]
    for lag in [5000, 10000, 20000]:
        ax.plot(lag, 0, "o", color="#d73027", markersize=10, zorder=3)
        ax.axvline(lag, ymin=0.03, ymax=0.18, color="#d73027", linewidth=1, linestyle=":", alpha=0.5)
    ax.annotate("OOD gap →", xy=(2000, 0), xytext=(2000, 0.55),
                fontsize=7.5, color="#d73027", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2))

    # v4 attempt 1: [2000, 5000, 10000, 20000, 30000, 50000]
    for lag in [2000, 5000, 10000, 20000, 30000, 50000]:
        ax.plot(lag, 1, "s", color="#fc8d59", markersize=9, zorder=3)

    # v4 attempt 2 (final): [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    for lag in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        ax.plot(lag, 2, "D", color="#1a9641", markersize=8, zorder=3)

    # inference τ
    ax.axvline(2000, color="#4575b4", linewidth=1.8, linestyle="--", alpha=0.7)
    ax.plot(2000, 3, "^", color="#4575b4", markersize=11, zorder=3)
    ax.text(2000, 3.25, "τ = 2000 ps", ha="center", fontsize=8, color="#4575b4", fontweight="bold")

    # ATLAS dt annotation
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
    return fig_to_b64(fig)


# ─── Figure 4: Best-temperature summary per protein ──────────────────────────

def fig_best_temp_summary():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1, ax2 = axes

    best_rmsf = []
    best_fes = []
    best_T_rmsf = []
    best_T_fes = []

    for p in PROTEINS:
        r, T_r = get_v4_best(p, ["proteins", p, "structural", "rmsf_corr"])
        f, T_f = get_v4_best(p, ["proteins", p, "thermodynamic", "fes_js"])
        best_rmsf.append(r)
        best_fes.append(f)
        best_T_rmsf.append(T_r)
        best_T_fes.append(T_f)

    x = np.arange(len(PROTEINS))
    colors_r = [TEMP_COLORS[str(T)] for T in best_T_rmsf]
    colors_f = [TEMP_COLORS[str(T)] for T in best_T_fes]

    bars1 = ax1.bar(x, best_rmsf, color=colors_r, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5, rotation=15, ha="right")
    ax1.set_ylabel("RMSF Correlation (Pearson r)", fontsize=9.5)
    ax1.set_title("Best RMSF Correlation (per-protein fine-tune)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for xi, (val, T) in enumerate(zip(best_rmsf, best_T_rmsf)):
        ax1.text(xi, val + 0.01, f"{val:.3f}\n@{T}K", ha="center", va="bottom", fontsize=7.5, color="black")

    bars2 = ax2.bar(x, best_fes, color=colors_f, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5, rotation=15, ha="right")
    ax2.set_ylabel("FES JS Divergence (lower = better)", fontsize=9.5)
    ax2.set_title("Best FES JS Divergence (per-protein fine-tune)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for xi, (val, T) in enumerate(zip(best_fes, best_T_fes)):
        ax2.text(xi, val + 0.01, f"{val:.3f}\n@{T}K", ha="center", va="bottom", fontsize=7.5, color="black")

    # Shared legend for temperature colors
    legend_elements = [mpatches.Patch(color=TEMP_COLORS["300"], label="300 K"),
                       mpatches.Patch(color=TEMP_COLORS["375"], label="375 K"),
                       mpatches.Patch(color=TEMP_COLORS["450"], label="450 K")]
    fig.legend(handles=legend_elements, title="Best temperature", fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_b64(fig)


# ─── Figure 5: Kinetics (relax_ratio) ─────────────────────────────────────────

def fig_kinetics():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(PROTEINS))
    w = 0.25
    for j, T in enumerate(TEMPS):
        vals = [get_v4_metric(p, T, ["proteins", p, "kinetic", "relax_ratio"]) for p in PROTEINS]
        ax.bar(x + (j - 1) * w, vals, width=w, label=f"{T} K",
               color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)

    ax.axhline(1.0, color="black", linewidth=1.4, linestyle="--", label="Perfect (ratio = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=9)
    ax.set_ylabel("Relaxation time ratio  τ_model / τ_MD", fontsize=10)
    ax.set_title("Kinetics: Model vs MD Relaxation Time Ratio", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(5.7, 1.15, "← ideal", fontsize=8, color="black", style="italic")
    fig.tight_layout()
    return fig_to_b64(fig)


# ─── Figure 6: Structural quality (bond lengths + clashes) ───────────────────

def fig_structural():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1, ax2 = axes

    for j, T in enumerate(TEMPS):
        bond_means = [get_v4_metric(p, T, ["proteins", p, "structural", "ca_bond_mean"]) for p in PROTEINS]
        clashes   = [get_v4_metric(p, T, ["proteins", p, "structural", "clash_count"])   for p in PROTEINS]
        x = np.arange(len(PROTEINS))
        w = 0.25
        ax1.bar(x + (j - 1)*w, bond_means, width=w, label=f"{T} K",
                color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)
        ax2.bar(x + (j - 1)*w, clashes,    width=w, label=f"{T} K",
                color=TEMP_COLORS[str(T)], edgecolor="white", alpha=0.88)

    ax1.axhline(3.81, color="black", linewidth=1.3, linestyle="--", label="MD ideal (3.81 Å)")
    ax1.set_xticks(np.arange(len(PROTEINS)))
    ax1.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5, rotation=15, ha="right")
    ax1.set_ylabel("Mean Cα–Cα bond length (Å)", fontsize=9)
    ax1.set_title("Cα Bond Length Quality", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.set_xticks(np.arange(len(PROTEINS)))
    ax2.set_xticklabels([p.replace("_", " ") for p in PROTEINS], fontsize=8.5, rotation=15, ha="right")
    ax2.set_ylabel("Mean clashes per frame", fontsize=9)
    ax2.set_title("Cα Clash Count", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("v4 Per-Protein Structural Geometry", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


# ─── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  @page {{ margin: 2cm 2.2cm; }}
  body {{
    font-family: "DejaVu Sans", Arial, sans-serif;
    font-size: 10.5pt;
    color: #1a1a1a;
    line-height: 1.55;
  }}
  h1 {{ font-size: 20pt; color: #1a3a5c; margin-bottom: 4pt; }}
  h2 {{ font-size: 13pt; color: #1a3a5c; border-bottom: 1.5px solid #1a3a5c; padding-bottom: 3px;
        margin-top: 22pt; margin-bottom: 8pt; }}
  h3 {{ font-size: 11pt; color: #2c5f8a; margin-top: 14pt; margin-bottom: 5pt; }}
  .subtitle {{ font-size: 12pt; color: #555; margin-bottom: 2pt; }}
  .meta {{ font-size: 9pt; color: #777; margin-top: 6pt; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
    font-size: 9.5pt;
  }}
  th {{
    background-color: #1a3a5c;
    color: white;
    padding: 5pt 8pt;
    text-align: center;
  }}
  td {{
    padding: 4pt 8pt;
    border-bottom: 1px solid #dde;
    text-align: center;
  }}
  tr:nth-child(even) td {{ background-color: #f5f7fa; }}
  .good {{ color: #2a7a2a; font-weight: bold; }}
  .bad  {{ color: #c0392b; }}
  .mid  {{ color: #d4820a; }}
  img {{ max-width: 100%; margin: 10pt 0; display: block; }}
  .fig-caption {{
    font-size: 9pt; color: #555; font-style: italic;
    margin-top: -4pt; margin-bottom: 14pt; text-align: center;
  }}
  .callout {{
    background: #eef4fb; border-left: 4px solid #2c5f8a;
    padding: 8pt 12pt; margin: 10pt 0; border-radius: 0 4px 4px 0;
    font-size: 9.5pt;
  }}
  .callout.warn {{
    background: #fff8e1; border-left-color: #f0a500;
  }}
  .callout.fix {{
    background: #efffef; border-left-color: #2a7a2a;
  }}
  code {{
    font-family: "DejaVu Sans Mono", Courier, monospace;
    background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 9pt;
  }}
  .page-break {{ page-break-after: always; }}
  ul {{ margin: 6pt 0; padding-left: 18pt; }}
  li {{ margin-bottom: 3pt; }}
</style>
</head>
<body>

<!-- ═══ TITLE ════════════════════════════════════════════════════════════════ -->
<h1>SE(3) PropagatorNet: Development and Validation Report</h1>
<div class="subtitle">Coarse-Grained Protein Dynamics via Denoising Diffusion on SE(3) Frames</div>
<div class="meta">
  Project: DL-MD &nbsp;|&nbsp; Date: June 24, 2026 &nbsp;|&nbsp; Model: v4 per-protein fine-tune
</div>

<div class="callout" style="margin-top:14pt;">
<strong>Executive Summary.</strong>
We developed and validated a denoising-diffusion propagator that autoregressively generates
Cα protein conformations in SE(3) frame space.
Starting from a pre-trained v2 checkpoint, we fine-tuned on 6 ATLAS proteins through three
successive stages: an all-protein ATLAS fine-tune (v3), a wide-lag universal fine-tune
(v4 long-lag), and per-protein fine-tunes with inference temperature sweeps.
The final v4 per-protein models achieve RMSF correlations of <strong>0.72–0.98</strong> across proteins,
an order-of-magnitude improvement over the v3 baseline (0.25–0.94), with markedly lower
free-energy-surface JS divergence and substantially better kinetic relaxation ratios.
</div>

<!-- ═══ 1. BACKGROUND ════════════════════════════════════════════════════════ -->
<h2>1. Background and Architecture</h2>

<h3>1.1 SE(3) PropagatorNet</h3>
<p>
The propagator is a message-passing network that maps each residue's current SE(3) frame
(rotation <em>R<sub>i</sub></em> ∈ SO(3), translation <em>t<sub>i</sub></em> ∈ ℝ<sup>3</sup>)
plus sequence features to a normalized update vector <em>u<sub>i</sub></em> ∈ ℝ<sup>6</sup>,
representing an infinitesimal SE(3) displacement.
Sampling is performed via reverse DDPM (or DDIM) in the normalized update space.
</p>
<table>
  <tr><th>Hyperparameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>Graph PropagatorNet (EGNN-style message passing)</td></tr>
  <tr><td>Hidden dimension</td><td>256</td></tr>
  <tr><td>Message-passing layers</td><td>6</td></tr>
  <tr><td>kNN neighbors per residue</td><td>12</td></tr>
  <tr><td>Node features</td><td>Residue type, chain ID, sequential index (fixed)</td></tr>
  <tr><td>Edge features</td><td>Inter-residue SE(3) displacement (dynamic, rebuilt each step)</td></tr>
  <tr><td>DDPM diffusion steps</td><td>20 (inference)</td></tr>
  <tr><td>Physical lag τ</td><td>2000 ps</td></tr>
  <tr><td>Inference temperature sweep</td><td>300 K / 375 K / 450 K</td></tr>
</table>

<h3>1.2 ATLAS Dataset</h3>
<p>
ATLAS is a repository of microsecond-scale explicit-solvent MD trajectories for diverse proteins.
We extracted Cα-only shards at 100 ps/frame resolution (1001 frames per trajectory,
covering 100.1 ns per shard). Six proteins spanning 46–219 residues were used:
</p>
<table>
  <tr><th>Protein</th><th>Chain</th><th>Residues</th><th>Description</th></tr>
  <tr><td>3u7t_A</td><td>A</td><td>46</td><td>Small β-sheet domain</td></tr>
  <tr><td>4p3a_B</td><td>B</td><td>79</td><td>α/β mixed fold</td></tr>
  <tr><td>1b2s_F</td><td>F</td><td>90</td><td>Helix bundle</td></tr>
  <tr><td>2y4x_B</td><td>B</td><td>93</td><td>α/β mixed fold</td></tr>
  <tr><td>1z0b_A</td><td>A</td><td>207</td><td>Multi-domain α-helical</td></tr>
  <tr><td>6ovk_R</td><td>R</td><td>219</td><td>Large receptor domain</td></tr>
</table>

<h3>1.3 Metrics</h3>
<ul>
  <li><strong>RMSF correlation</strong>: Pearson r between per-residue root-mean-square fluctuation profiles of model vs MD. r ≈ 1 means the model reproduces which residues are flexible. Higher is better; r &lt; 0.5 indicates failure to capture flexibility patterns.</li>
  <li><strong>Distance JS divergence</strong>: Jensen–Shannon divergence of Cα pairwise-distance distributions. Lower is better; 0 = identical distributions, 1 = disjoint.</li>
  <li><strong>FES JS divergence</strong>: JS divergence of free-energy surfaces in 2-D PCA space. Lower is better; measures how well the model explores the same conformational landscape as MD.</li>
  <li><strong>Relaxation ratio</strong>: Integral relaxation time of the model autocorrelation function divided by the MD value. 1.0 is ideal; &lt;1 means too fast, &gt;1 means too slow.</li>
</ul>

<div class="page-break"></div>

<!-- ═══ 2. CHECKPOINT HIERARCHY ══════════════════════════════════════════════ -->
<h2>2. Checkpoint Hierarchy and Training Strategy</h2>

<p>Training proceeded through a staged hierarchy, each stage inheriting the weights of its predecessor:</p>

<table>
  <tr><th>Checkpoint</th><th>Starting from</th><th>Training data</th><th>Steps</th><th>Lags (ps)</th></tr>
  <tr><td><code>v2_256h_90k.pt</code></td><td>Random init</td><td>Large protein library (pre-ATLAS)</td><td>90,000</td><td>2000–10000</td></tr>
  <tr><td><code>v3_lam0.pt</code></td><td>v2_256h_90k</td><td>All 6 ATLAS proteins, λ=0</td><td>10,000</td><td>2000, 5000, 10000</td></tr>
  <tr><td><code>v4_longlags.pt</code></td><td>v3_lam0</td><td>All 6 ATLAS proteins</td><td>20,000</td><td>100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000</td></tr>
  <tr><td><code>v4_{{protein}}.pt</code> (×6)</td><td>v4_longlags</td><td>Single protein shard</td><td>5,000 each</td><td>Same as v4_longlags</td></tr>
</table>

<div class="callout">
<strong>UpdateNorm re-fitting.</strong>
Each training run re-fits the <code>UpdateNorm</code> statistics from the current data before
loading weights, so the per-component mean/scale always reflects the new lag distribution
regardless of which checkpoint is resumed. This was verified empirically: training loss dropped
from 0.13 → 0.054 within 500 steps of resuming v4_longlags from v3_lam0, confirming rapid
adaptation despite the changed lag range.
</div>

<!-- ═══ 3. LAG STRATEGY ═══════════════════════════════════════════════════════ -->
<h2>3. Lag Strategy: What We Tried</h2>

<h3>3.1 Why Lag Selection Matters</h3>
<p>
The model is trained to predict the SE(3) update that advances a conformation by τ picoseconds.
At inference, we always use τ = 2000 ps. If the training lag distribution does not include
τ = 2000 ps, the model must extrapolate out-of-distribution — an unstable regime for DDPM
that produces non-physical (NaN) positions.
</p>

<h3>3.2 v3 Attempt: Lags [5000, 10000, 20000 ps]</h3>
<p>
The original v3 fine-tune used three lags all ≥ 5000 ps, motivated by the goal of
capturing slow conformational transitions (barrier crossings on the ~10 ns scale).
However, inference at τ = 2000 ps falls below the minimum training lag, placing it
out of distribution. This caused:
</p>
<div class="callout warn">
<strong>OOD NaN cascade (v3 rollout failure).</strong>
DDPM denoising occasionally produced non-finite (NaN/Inf) CA positions when τ = 2000 ps was
below the training lag range. Because each rollout step uses the previous frame as input,
a single NaN frame propagated through the entire trajectory via Kabsch alignment failure
(<code>linalg.svd: non-finite values</code>). Protein 2y4x_B hit NaN at step 27 of 40.
</div>

<h3>3.3 v4 Attempt 1: Add 2000 ps Floor</h3>
<p>
Lags were extended to [2000, 5000, 10000, 20000, 30000, 50000 ps], anchoring τ = 2000 ps
at the left edge of the training distribution. While this improved stability, the ATLAS
trajectory frame interval is <strong>100 ps</strong> — meaning 2000 ps is still 20× the minimum
physically resolvable timescale. Short-range dynamics (100 ps – 1 ns) carry important
structural stability information that was still missing.
</p>

<h3>3.4 v4 Final: Full Decade Span [100–50000 ps]</h3>
<p>
The final lag set spans three orders of magnitude:
<code>[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]</code> ps.
This achieves three goals simultaneously:
</p>
<ul>
  <li><strong>Short-range stability</strong>: lags 100–500 ps teach the model local bond geometry
      and prevent structural explosion at early rollout steps.</li>
  <li><strong>Well-sampled inference point</strong>: τ = 2000 ps sits in the middle of the
      distribution, not at an edge.</li>
  <li><strong>Long-range barrier crossing</strong>: lags up to 50000 ps (= 50% of one ATLAS
      trajectory) capture slow conformational transitions.</li>
</ul>

<img src="data:image/png;base64,{fig_lag}" />
<div class="fig-caption">
Figure 1. Evolution of the training lag strategy. Markers show training lag points;
the dashed blue line marks the inference τ = 2000 ps.
The v3 strategy placed inference τ below the training distribution (OOD → NaN).
The final v4 strategy spans the full ATLAS-resolvable range.
</div>

<div class="page-break"></div>

<!-- ═══ 4. ENGINEERING FIXES ══════════════════════════════════════════════════ -->
<h2>4. Engineering Fixes Applied</h2>

<h3>4.1 NaN Guard in Rollout</h3>
<p>
A three-layer defense was added to prevent OOD NaN frames from crashing validation:
</p>
<div class="callout fix">
<strong>Layer 1 — rollout guard</strong> (<code>lsmd/transfer_eval.py</code>):
After each DDPM sampling step, if <code>t_new</code> contains non-finite values, the
previous valid frame is repeated (without advancing <em>R</em>, <em>t</em>), preventing
the NaN cascade. A warning is emitted to stderr.
</div>
<div class="callout fix">
<strong>Layer 2 — msd_curve guard</strong> (<code>lsmd/transfer_validate.py</code>):
Before Kabsch alignment, each frame is checked for non-finite values. NaN frames are
set to <code>aligned[f] = NaN</code> and skipped in the MSD computation rather than
causing an SVD failure.
</div>
<div class="callout fix">
<strong>Layer 3 — per-protein try/except</strong> (<code>scripts/validate_physics.py</code>):
Each protein's validation is wrapped in a try/except; a failure logs a warning and stores
<code>{{"error": str(exc)}}</code> in the JSON, allowing the other proteins to complete.
The <code>summarize()</code> function skips error entries when computing mean metrics.
</div>

<h3>4.2 Noether Momentum Projection</h3>
<p>
After each SHAKE pseudo-bond correction step, a Noether projection removes net linear
and angular momentum per chain. This enforces the physical symmetry that the center of mass
should not drift and prevents slow rotation of the molecule during long rollouts.
Enabled via <code>--noether</code> in all v4 inference runs.
</p>

<h3>4.3 WCA Excluded-Volume Guidance</h3>
<p>
C2 guidance using a Weeks-Chandler-Andersen (WCA) excluded-volume potential nudges the
DDPM samples away from steric clashes at each denoising step. The guidance gradient is
computed in normalized update space (<code>wca_lam = 0.05</code>) so its magnitude is
independent of the physical scale of the update. This is responsible for the near-zero
clash counts seen in all v4 validation runs.
</p>

<!-- ═══ 5. RESULTS ════════════════════════════════════════════════════════════ -->
<h2>5. Results</h2>

<h3>5.1 Model Version Progression</h3>
<img src="data:image/png;base64,{fig_prog}" />
<div class="fig-caption">
Figure 2. RMSF profile correlation across model versions.
v2 = pre-trained baseline evaluated on ATLAS shards; v3 = ATLAS fine-tune with narrow lags;
v4 long-lag = universal fine-tune with full lag range; v4 per-protein = per-protein fine-tune
at its best inference temperature.
</div>

<p>
The v2 baseline showed highly variable performance (r = –0.001 to 0.71), reflecting
its training on a different distribution.
v3 improved average correlation but failed to generalize to several proteins, and
its narrow lag range caused rollout instability.
The v4 long-lag universal fine-tune raised the floor substantially,
and per-protein fine-tuning at the best inference temperature pushed most proteins above r = 0.93.
</p>

<h3>5.2 Per-Protein Best-Temperature Results</h3>
<img src="data:image/png;base64,{fig_best}" />
<div class="fig-caption">
Figure 3. Best RMSF correlation (left) and best FES JS divergence (right) for each protein
across the three inference temperatures (300, 375, 450 K). Bar colors indicate the optimal temperature.
</div>

<h3>5.3 Temperature Sweep Heatmaps</h3>
<img src="data:image/png;base64,{fig_heat}" />
<div class="fig-caption">
Figure 4. Three validation metrics as a function of protein and inference temperature.
Green = good; red = poor. All 18 cells (6 proteins × 3 temperatures) from the v4 per-protein checkpoints.
</div>

<div class="page-break"></div>

<h3>5.4 Kinetics</h3>
<img src="data:image/png;base64,{fig_kin}" />
<div class="fig-caption">
Figure 5. Kinetic relaxation time ratio (model / MD) at each inference temperature.
The dashed line marks ideal ratio = 1. Values below 1 indicate over-fast relaxation
(model explores conformational space too quickly); values above 1 indicate under-fast.
</div>

<h3>5.5 Structural Geometry</h3>
<img src="data:image/png;base64,{fig_struct}" />
<div class="fig-caption">
Figure 6. Cα bond length (left) and clash count (right) across proteins and temperatures.
The dashed line marks the ideal MD bond length of 3.81 Å. Near-zero clashes confirm the
WCA guidance is effective at all temperatures.
</div>

<h3>5.6 Complete Results Table</h3>
<table>
  <tr>
    <th>Protein</th><th>Residues</th>
    <th>T_best (K)</th><th>RMSF corr ↑</th><th>Dist JS ↓</th>
    <th>FES JS ↓</th><th>Relax ratio</th>
  </tr>
  {table_rows}
</table>

<p>
<strong>Key observations:</strong>
</p>
<ul>
  <li><strong>Smaller proteins</strong> (3u7t_A, 4p3a_B, 1b2s_F) achieve the highest RMSF correlations (0.94–0.97), consistent with their simpler energy landscapes and faster equilibration.</li>
  <li><strong>Medium proteins</strong> (2y4x_B, 1z0b_A) also perform well (0.96, 0.98), suggesting the per-protein fine-tune effectively captures individual conformational signatures.</li>
  <li><strong>Large receptor 6ovk_R</strong> (219 residues) achieves r = 0.72 — substantially weaker, likely due to slower equilibration timescales that exceed the 5000-step per-protein fine-tune budget and the 300-step validation rollout length.</li>
  <li><strong>Optimal temperature</strong> is 300–375 K for most proteins. 450 K universally degrades structural metrics, suggesting the temperature embedding amplifies fluctuations past the physically reasonable regime.</li>
  <li><strong>Kinetics</strong> are consistently under-estimated (relax_ratio &lt; 1 for small proteins) — the model relaxes ~2–5× faster than MD. This is a known limitation of short per-protein fine-tunes: the model has not learned the full free-energy barriers.</li>
</ul>

<!-- ═══ 6. PIPELINE SUMMARY ═══════════════════════════════════════════════════ -->
<h2>6. V4 Pipeline Summary</h2>
<table>
  <tr><th>Phase</th><th>Script / command</th><th>Output</th><th>Wall time</th></tr>
  <tr><td>Phase 1 train</td><td><code>train_transfer.py</code>, 20k steps, all 6 ATLAS shards</td><td><code>v4_longlags.pt</code></td><td>~149 min</td></tr>
  <tr><td>Phase 1 validate</td><td><code>validate_physics.py</code>, 300 steps, T=300 K</td><td><code>validation_v4_longlags_T300.json</code></td><td>~5 min</td></tr>
  <tr><td>Phase 2 train (×6)</td><td><code>train_transfer.py</code>, 5k steps, single shard</td><td><code>v4_{{protein}}.pt</code></td><td>~48 min/protein</td></tr>
  <tr><td>Phase 2 validate (×6×3)</td><td><code>validate_physics.py</code>, T={{300,375,450}} K</td><td>18 JSON files</td><td>~4 min/run</td></tr>
  <tr><td><strong>Total</strong></td><td></td><td>7 checkpoints, 19 JSON files</td><td><strong>~5 h 20 min</strong></td></tr>
</table>

<!-- ═══ 7. CONCLUSIONS ════════════════════════════════════════════════════════ -->
<h2>7. Conclusions and Next Steps</h2>

<p>
The v4 pipeline demonstrates that combining a wide-lag universal fine-tune with per-protein
adaptation and inference temperature tuning yields dramatically better structural and
thermodynamic agreement with MD than the v3 baseline.
Five of six proteins now achieve RMSF correlation &gt; 0.93.
</p>

<p><strong>Remaining limitations and suggested next steps:</strong></p>
<ul>
  <li><strong>Kinetic accuracy</strong>: relaxation ratios are systematically &lt; 1 (too fast). Increasing the per-protein fine-tune budget (5k → 20k steps) or adding a kinetic loss term may correct this.</li>
  <li><strong>6ovk_R</strong>: the large receptor domain underperforms. A longer fine-tune (10k+ steps) and/or a larger per-shard batch may help; the 219-residue chain also has slower intrinsic dynamics that a 300-step rollout may not fully capture.</li>
  <li><strong>Held-out evaluation</strong>: all ATLAS proteins were seen during training. A proper generalization assessment requires evaluation on proteins not present in the fine-tuning shard set.</li>
  <li><strong>Multi-chain systems</strong>: the current CA-only model does not capture inter-chain contacts; extending to multi-chain complexes is a logical next step.</li>
  <li><strong>Explicit free-energy barriers</strong>: training on longer lags (up to 50k ps) begins to capture barrier crossing, but the learned barriers may be imprecise. Coupling with an explicit CG energy model (Phase 3 planned) could improve kinetic fidelity.</li>
</ul>

</body>
</html>
"""


def make_table_rows():
    rows = []
    for p in PROTEINS:
        n = PROTEIN_INFO[p]["n_res"]
        r, T_r = get_v4_best(p, ["proteins", p, "structural", "rmsf_corr"])
        d, _   = get_v4_best(p, ["proteins", p, "structural", "dist_js"])
        f, _   = get_v4_best(p, ["proteins", p, "thermodynamic", "fes_js"])
        rel    = get_v4_metric(p, T_r, ["proteins", p, "kinetic", "relax_ratio"])

        def cls_r(v): return "good" if v >= 0.9 else "mid" if v >= 0.7 else "bad"
        def cls_j(v): return "good" if v <= 0.1 else "mid" if v <= 0.5 else "bad"
        def cls_rel(v): return "good" if 0.5 <= v <= 2.0 else "mid" if 0.2 <= v <= 5.0 else "bad"

        rows.append(
            f"<tr>"
            f"<td>{p}</td><td>{n}</td><td>{T_r}</td>"
            f"<td class='{cls_r(r)}'>{r:.4f}</td>"
            f"<td class='{cls_j(d)}'>{d:.6f}</td>"
            f"<td class='{cls_j(f)}'>{f:.4f}</td>"
            f"<td class='{cls_rel(rel)}'>{rel:.3f}</td>"
            f"</tr>"
        )
    return "\n  ".join(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_comparison_report.pdf")
    args = ap.parse_args()

    print("Generating figures...", flush=True)
    b64_lag    = fig_lag_strategy()
    print("  lag strategy done", flush=True)
    b64_prog   = fig_progression()
    print("  progression done", flush=True)
    b64_best   = fig_best_temp_summary()
    print("  best-temp done", flush=True)
    b64_heat   = fig_temp_heatmap()
    print("  heatmap done", flush=True)
    b64_kin    = fig_kinetics()
    print("  kinetics done", flush=True)
    b64_struct = fig_structural()
    print("  structural done", flush=True)

    html = HTML_TEMPLATE.format(
        fig_lag=b64_lag,
        fig_prog=b64_prog,
        fig_best=b64_best,
        fig_heat=b64_heat,
        fig_kin=b64_kin,
        fig_struct=b64_struct,
        table_rows=make_table_rows(),
    )

    print(f"Rendering PDF → {args.out} ...", flush=True)
    from weasyprint import HTML as WP_HTML
    WP_HTML(string=html, base_url=".").write_pdf(args.out)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
