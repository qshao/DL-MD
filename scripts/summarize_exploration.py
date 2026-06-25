"""Post-MD analysis: read a completed summary.json and report validated structures.

After running explore_conformations.py and populating md_pass/md_rmsd_final/
md_rg_final in summary.json, run this script to print a classification table
and save a CV-space plot of survivors.

Usage
-----
python scripts/summarize_exploration.py --out explore_out/3u7t_A
"""
import argparse
import json
import os

import numpy as np


_CLASSIFY = [
    (3.0, "Alternative state (>3 Å from native)"),
    (1.0, "Expanded fluctuation (1-3 Å)"),
    (0.0, "Near-native (<1 Å)"),
]


def classify_rmsd(rmsd):
    for threshold, label in _CLASSIFY:
        if rmsd > threshold:
            return label
    return "Near-native (<1 Å)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="explore_out directory")
    args = ap.parse_args()

    summary_path = os.path.join(args.out, "summary.json")
    cv_path = os.path.join(args.out, "cv_coords.npy")

    with open(summary_path) as fh:
        records = json.load(fh)

    total = len(records)
    validated = [r for r in records if r.get("md_pass") is True]
    pending = [r for r in records if r.get("md_pass") is None]

    print(f"\n=== Exploration Summary ({args.out}) ===")
    print(f"Total accepted (geometry filter):  {total}")
    print(f"MD-validated (md_pass=True):       {len(validated)}")
    print(f"MD-rejected  (md_pass=False):      {total - len(validated) - len(pending)}")
    print(f"Pending MD:                        {len(pending)}")

    if validated:
        print("\nMD-validated structures:")
        print(f"{'ID':>6}  {'RMSD_native(Å)':>15}  {'MD_RMSD(Å)':>11}  Classification")
        print("-" * 65)
        for r in sorted(validated, key=lambda x: x["rmsd_native"], reverse=True):
            cls = classify_rmsd(r["rmsd_native"])
            rmsd_md = r["md_rmsd_final"] if r["md_rmsd_final"] is not None else "N/A"
            print(f"{r['id']:>6}  {r['rmsd_native']:>15.3f}  {str(rmsd_md):>11}  {cls}")

        # Category counts
        print("\nClassification breakdown:")
        for threshold, label in _CLASSIFY:
            count = sum(1 for r in validated if r["rmsd_native"] > threshold)
            print(f"  {label}: {count}")

    # Plot if cv_coords available
    if os.path.exists(cv_path) and validated:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cv_all = np.load(cv_path)                 # [M, n_cv]
            ids_all = [r["id"] for r in records]
            id_to_idx = {rid: i for i, rid in enumerate(ids_all)}

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(cv_all[:, 0], cv_all[:, 1],
                       c="lightgrey", s=15, label="geometry-passed", zorder=1)

            colors = {"Alternative state (>3 Å from native)": "red",
                      "Expanded fluctuation (1-3 Å)": "orange",
                      "Near-native (<1 Å)": "green"}
            for r in validated:
                idx = id_to_idx.get(r["id"])
                if idx is None:
                    continue
                cls = classify_rmsd(r["rmsd_native"])
                ax.scatter(cv_all[idx, 0], cv_all[idx, 1],
                           c=colors.get(cls, "blue"), s=60, zorder=3,
                           edgecolors="black", linewidths=0.5)

            # Legend patches
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(color=c, label=l)
                       for l, c in colors.items()]
            handles.append(mpatches.Patch(color="lightgrey",
                                          label="geometry-passed (not MD-run)"))
            ax.legend(handles=handles, fontsize=8)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"MD-validated conformations — {len(validated)}/{total} survivors")
            plt.tight_layout()
            fig_path = os.path.join(args.out, "md_summary.png")
            plt.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"\nFigure saved: {fig_path}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
