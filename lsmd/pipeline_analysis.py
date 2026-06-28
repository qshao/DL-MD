"""Objective-specific analysis for the hybrid ML-MD pipeline."""
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pairwise_rmsd(ca_list):
    """Compute [M, M] pairwise RMSD matrix from list of [N, 3] Å arrays."""
    M = len(ca_list)
    mat = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            diff = ca_list[i] - ca_list[j]
            rmsd = float(np.sqrt((diff ** 2).sum(-1).mean()))
            mat[i, j] = mat[j, i] = rmsd
    return mat


def _medoid(indices, rmsd_matrix):
    """Return the medoid index (min sum of distances within cluster)."""
    sub = rmsd_matrix[np.ix_(indices, indices)]
    return indices[int(sub.sum(axis=1).argmin())]


def _cluster_structures(ca_list, rmsd_cutoff_A):
    """Ward hierarchical clustering on Cα RMSD.

    Returns:
        labels: [M] int array, cluster index per structure (1-based)
        rmsd_matrix: [M, M] float32 pairwise RMSD matrix in Å
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    M = len(ca_list)
    if M == 0:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=np.float32)
    if M == 1:
        return np.array([1], dtype=int), np.zeros((1, 1), dtype=np.float32)
    mat = _pairwise_rmsd(ca_list)
    condensed = squareform(mat, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=rmsd_cutoff_A, criterion="distance")
    return labels, mat


def _load_stable_ca_frames(md_runs_dir):
    """Load the final Cα frame from each stable MD run.

    Returns:
        frames: list of [N, 3] Å arrays
        ids:    list of run_id strings in the same order
    """
    import mdtraj as md

    frames, ids = [], []
    for run_id in sorted(os.listdir(md_runs_dir)):
        metrics_path = os.path.join(md_runs_dir, run_id, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if not m.get("stable", False):
            continue
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if not os.path.exists(traj_path):
            continue
        traj   = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_nm  = traj.atom_slice(ca_idx)[-1].xyz[0]   # [N, 3] nm
        frames.append(ca_nm * 10.0)                    # → Å
        ids.append(run_id)
    return frames, ids


# ---------------------------------------------------------------------------
# Explore: diverse stable library
# ---------------------------------------------------------------------------

def analyze_explore(md_runs_dir, out_dir, rmsd_cutoff_A=2.0):
    """Cluster stable MD structures into a diverse library.

    Args:
        md_runs_dir (str): Directory containing per-structure MD run subdirs.
        out_dir (str):     Output directory for library/ and cluster_summary.json.
        rmsd_cutoff_A (float): Ward clustering distance cutoff in Å (default 2.0).

    Returns:
        dict: n_proposals_attempted, n_stable, n_clusters, representatives list.
    """
    import mdtraj as md

    os.makedirs(out_dir, exist_ok=True)
    lib_dir = os.path.join(out_dir, "library")
    os.makedirs(lib_dir, exist_ok=True)

    # Count total runs attempted
    all_run_ids = [d for d in os.listdir(md_runs_dir)
                   if os.path.exists(os.path.join(md_runs_dir, d, "metrics.json"))]
    n_attempted = len(all_run_ids)

    frames, stable_ids = _load_stable_ca_frames(md_runs_dir)
    n_stable = len(frames)

    representatives = []
    if n_stable == 0:
        summary = {
            "n_proposals_attempted": n_attempted,
            "n_stable": 0,
            "n_clusters": 0,
            "rmsd_cutoff_A": rmsd_cutoff_A,
            "representatives": [],
        }
        with open(os.path.join(out_dir, "cluster_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    labels, rmsd_matrix = _cluster_structures(frames, rmsd_cutoff_A)
    unique_labels = sorted(set(labels))

    for cl in unique_labels:
        members = np.where(labels == cl)[0]
        med_idx = _medoid(members, rmsd_matrix)
        run_id  = stable_ids[med_idx]

        # Load all-atom PDB of medoid and copy to library
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if os.path.exists(traj_path):
            traj = md.load(traj_path, top=top_path)
            traj[-1].save_pdb(os.path.join(lib_dir, f"cluster{cl:04d}_{run_id}.pdb"))

        representatives.append({
            "cluster_id": int(cl),
            "size": int(len(members)),
            "medoid_id": run_id,
        })

    summary = {
        "n_proposals_attempted": n_attempted,
        "n_stable": n_stable,
        "n_clusters": len(unique_labels),
        "rmsd_cutoff_A": rmsd_cutoff_A,
        "representatives": representatives,
    }
    with open(os.path.join(out_dir, "cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"explore: {n_stable}/{n_attempted} stable → {len(unique_labels)} clusters")
    return summary


# ---------------------------------------------------------------------------
# Helpers for FES and kinetics: load all frames from stable runs
# ---------------------------------------------------------------------------

def _load_all_ca_frames(md_runs_dir, n_frames_per_run=None):
    """Load all Cα frames from every stable MD run.

    Returns:
        all_frames: torch.Tensor [T, N, 3] in Å
        n_frames:   int — total frames loaded
    """
    import torch
    import mdtraj as md

    frame_list = []
    for run_id in sorted(os.listdir(md_runs_dir)):
        metrics_path = os.path.join(md_runs_dir, run_id, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if not m.get("stable", False):
            continue
        traj_path = os.path.join(md_runs_dir, run_id, "trajectory.dcd")
        top_path  = os.path.join(md_runs_dir, run_id, "topology.pdb")
        if not os.path.exists(traj_path):
            continue
        traj   = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_nm  = traj.atom_slice(ca_idx).xyz          # [F, N, 3] nm
        ca_A   = torch.tensor(ca_nm * 10.0)           # → Å
        if n_frames_per_run is not None:
            ca_A = ca_A[:n_frames_per_run]
        frame_list.append(ca_A)

    if not frame_list:
        return torch.zeros(0), 0
    all_frames = torch.cat(frame_list, dim=0)   # [T, N, 3]
    return all_frames, all_frames.shape[0]


# ---------------------------------------------------------------------------
# FES: free energy surface over CV space
# ---------------------------------------------------------------------------

_kB_KCAL = 0.001987   # kcal / (mol · K)


def analyze_fes(md_runs_dir, cv_basis_path, out_dir,
                temp_K=310.0, n_bins=50):
    """Estimate free energy surface by projecting MD frames onto CV space.

    Args:
        md_runs_dir (str):   Directory with per-run MD subdirs.
        cv_basis_path (str): Path to cv_basis.pt written by explore_conformations.py.
        out_dir (str):       Output directory.
        temp_K (float):      Temperature for Boltzmann inversion (default 310.0).
        n_bins (int):        Histogram bins per CV axis (default 50).

    Returns:
        dict: n_frames_total, n_frames_stable, temp_K, fes_min_kcal, fes_max_kcal, n_bins.
    """
    import torch
    from lsmd.cv_guidance import CVSpace

    os.makedirs(out_dir, exist_ok=True)

    cv_space = CVSpace.load(cv_basis_path)
    cv_space.to("cpu")

    all_frames, n_frames = _load_all_ca_frames(md_runs_dir)

    if n_frames == 0:
        print("fes: no stable frames found")
        summary = {"n_frames_total": 0, "n_frames_stable": 0, "fes_min_kcal": None, "fes_max_kcal": None,
                   "temp_K": temp_K, "n_bins": n_bins}
        with open(os.path.join(out_dir, "fes_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    # Project all frames onto first two CVs (PC1, PC2)
    with torch.no_grad():
        cv_all = cv_space.project_batch(all_frames)   # [T, n_cv+2]
    pc1 = cv_all[:, 0].numpy()
    pc2 = cv_all[:, 1].numpy()

    # 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        pc1, pc2, bins=n_bins, density=True
    )

    # Boltzmann inversion: F = -kT ln P, shift minimum to 0
    kT = _kB_KCAL * temp_K
    with np.errstate(divide="ignore"):
        fes = -kT * np.log(hist + 1e-12)
    fes -= fes.min()

    np.save(os.path.join(out_dir, "fes.npy"), fes.astype(np.float32))
    np.save(os.path.join(out_dir, "cv_edges.npy"),
            np.array([x_edges, y_edges], dtype=object))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        pcm = ax.pcolormesh(x_edges, y_edges, fes.T, cmap="viridis_r", vmin=0)
        plt.colorbar(pcm, ax=ax, label="FES (kcal/mol)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"FES — {n_frames} frames, T={temp_K} K")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fes.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    summary = {
        "n_frames_total": n_frames,
        "n_frames_stable": n_frames,
        "temp_K": temp_K,
        "n_bins": n_bins,
        "fes_min_kcal": round(float(fes.min()), 4),
        "fes_max_kcal": round(float(fes.max()), 4),
    }
    with open(os.path.join(out_dir, "fes_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"fes: {n_frames} frames → FES max {fes.max():.2f} kcal/mol")
    return summary
