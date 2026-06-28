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
