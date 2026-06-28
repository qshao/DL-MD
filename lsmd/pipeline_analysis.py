"""Objective-specific analysis for the hybrid ML-MD pipeline."""
import json
import os

import numpy as np

try:
    import pyemma  # noqa: F401
    _HAS_PYEMMA = True
except ImportError:
    _HAS_PYEMMA = False


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


# ---------------------------------------------------------------------------
# Kinetics: MSM construction via PyEMMA
# ---------------------------------------------------------------------------

def _load_featurised_trajs(md_runs_dir):
    """Load and featurise stable MD trajectories as Cα pairwise distances.

    Returns:
        list of [T_i, n_features] float32 arrays — one per stable run
    """
    import mdtraj as md

    trajs = []
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
        ca_traj = traj.atom_slice(ca_idx)
        n = ca_traj.n_atoms
        # All Cα pairs more than 3 residues apart
        pairs = [(i, j) for i in range(n) for j in range(i + 3, n)]
        if not pairs:
            continue
        dists = md.compute_distances(ca_traj, pairs)  # [T, n_pairs] in nm
        trajs.append(dists.astype(np.float32))
    return trajs


def analyze_kinetics(md_runs_dir, out_dir,
                     tica_lag=50, n_clusters=100, msm_lag=5):
    """Build a Markov State Model from stable MD trajectories.

    Args:
        md_runs_dir (str): Directory with per-run MD subdirs.
        out_dir (str):     Output directory.
        tica_lag (int):    TICA lag time in frames (1 frame = 10 ps; default 50 = 500 ps).
        n_clusters (int):  k-means cluster count (default 100).
        msm_lag (int):     MSM lag time in frames (default 5 = 50 ps).

    Returns:
        dict: n_trajectories, total_frames, n_states, implied_timescales_ns, etc.
    """
    if not _HAS_PYEMMA:
        raise ImportError(
            "pyemma is required: conda install -c conda-forge pyemma"
        )
    import pyemma

    os.makedirs(out_dir, exist_ok=True)

    trajs = _load_featurised_trajs(md_runs_dir)
    n_traj = len(trajs)
    if n_traj == 0:
        print("kinetics: no stable trajectories found")
        return {"n_trajectories": 0}

    total_frames = sum(t.shape[0] for t in trajs)

    # TICA
    tica = pyemma.coordinates.tica(trajs, lag=tica_lag, dim=5, kinetic_map=True)
    tica_output = tica.get_output()
    tica_coords = np.concatenate(tica_output, axis=0)  # [T_total, 5]
    np.save(os.path.join(out_dir, "tica_projection.npy"), tica_coords)

    # k-means clustering
    k = min(n_clusters, total_frames // 2)
    cluster = pyemma.cluster.kmeans(tica_output, k=k, max_iter=100, stride=1)
    np.save(os.path.join(out_dir, "state_assignments.npy"),
            np.concatenate(cluster.dtrajs))

    # MSM
    msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=msm_lag)
    np.save(os.path.join(out_dir, "transition_matrix.npy"),
            msm.transition_matrix.astype(np.float32))

    # Implied timescales (top 5 processes)
    its_frames = msm.timescales(k=min(5, k - 1))
    dt_ps = 10.0  # 1 frame = 10 ps (saved every 5000 steps at 2 fs)
    its_ns = (its_frames * msm_lag * dt_ps / 1000).tolist()

    timescales_path = os.path.join(out_dir, "timescales.json")
    with open(timescales_path, "w") as fh:
        json.dump({"implied_timescales_ns": its_ns, "msm_lag_frames": msm_lag}, fh, indent=2)

    # ITS plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(1, len(its_ns) + 1), its_ns)
        ax.set_xlabel("Process"); ax.set_ylabel("Implied timescale (ns)")
        ax.set_title(f"MSM implied timescales ({k} states, lag={msm_lag} frames)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "timescales.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    summary = {
        "n_trajectories": n_traj,
        "total_frames": total_frames,
        "n_states": k,
        "tica_lag_frames": tica_lag,
        "msm_lag_frames": msm_lag,
        "implied_timescales_ns": its_ns,
    }
    with open(os.path.join(out_dir, "msm_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"kinetics: {n_traj} trajs → {k} states, top ITS={its_ns[0]:.1f} ns")
    return summary
