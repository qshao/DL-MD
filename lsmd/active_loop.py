"""Active learning loop utilities for single-protein conformational exploration.

Provides:
  _pdb_to_shard      — load a static PDB into a 1-frame shard dict
  _geometry_pass_rate — fraction of proposals with good bonds and no clashes
  _min_rmsd_kabsch   — minimum Kabsch-aligned RMSD from one structure to many
  bootstrap_check    — decide zero-MD or short-MD starting shard
  shard_from_md_runs — extract (R, t) Cα frames from completed OpenMM MD runs
  build_replay_shard — combine new frames with replay buffer for fine-tuning
  check_convergence  — budget / coverage / fes stopping criterion
"""
import json
import os

import mdtraj as md
import numpy as np
import torch

from lsmd import data, geometry as g
from lsmd.cv_guidance import CVSpace
from lsmd.transfer_eval import load_checkpoint, rollout
from lsmd.vocab import residue_indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pdb_to_shard(pdb_path: str, dt_ps: float = 200.0) -> dict:
    """Load a static all-atom PDB and return a 1-frame shard dict.

    Uses mdtraj to extract protein backbone (N, CA, C) atoms, computes SE(3)
    rotation matrices via geometry.build_frames(), and assembles the full shard
    dict consumed by train_transfer.py and explore_conformations.py.

    Args:
        pdb_path: path to heavy-atom PDB (crystal structure or AlphaFold).
        dt_ps:    nominal frame spacing in ps (200 ps default).

    Returns:
        dict with keys:
            R         [1, N, 3, 3] float32 — per-residue rotation matrices
            t         [1, N, 3]    float32 — Cα positions in Å
            res_type  [N] long     — residue type indices (0-based)
            chain_id  [N] long     — chain indices (0-based)
            res_index [N] long     — sequential residue index
            n_res     int
            dt        float        — ps per frame
            seq       list[str]    — residue 3-letter names
    """
    traj = md.load(pdb_path)
    top  = traj.topology

    # Collect backbone atoms in residue order; skip non-protein residues
    residues = [r for r in top.residues if r.is_protein]
    n_idx, ca_idx, c_idx = [], [], []
    res_names, chain_ids = [], []
    for r in residues:
        atoms = {a.name: a.index for a in r.atoms}
        if not all(k in atoms for k in ("N", "CA", "C")):
            continue
        n_idx.append(atoms["N"])
        ca_idx.append(atoms["CA"])
        c_idx.append(atoms["C"])
        res_names.append(r.name)
        chain_ids.append(r.chain.index)

    xyz = torch.tensor(traj.xyz, dtype=torch.float32) * 10.0  # nm → Å  [1, n_atoms, 3]
    N_pos  = xyz[:, n_idx,  :]   # [1, N, 3]
    CA_pos = xyz[:, ca_idx, :]
    C_pos  = xyz[:, c_idx,  :]

    R, t = g.build_frames(N_pos, CA_pos, C_pos)  # [1, N, 3, 3], [1, N, 3]

    res_type  = residue_indices(res_names)
    chain_id  = torch.tensor(chain_ids, dtype=torch.long)
    res_index = torch.arange(len(res_names), dtype=torch.long)

    return {
        "R": R, "t": t,
        "res_type": res_type,
        "chain_id": chain_id,
        "res_index": res_index,
        "n_res": len(res_names),
        "dt": float(dt_ps),
        "seq": res_names,
    }


def _geometry_pass_rate(proposals: list, ref_bond_A: float,
                        bond_tol: float = 0.15, clash_dist: float = 3.5) -> float:
    """Fraction of [N,3] Cα proposals passing bond-length and clash checks.

    Bond check: every adjacent Cα–Cα bond within ±bond_tol Å of ref_bond_A.
    Clash check: all non-adjacent (|i-j|>1) Cα–Cα distances > clash_dist Å.

    Args:
        proposals:    list of [N, 3] float tensors in Å.
        ref_bond_A:   reference Cα–Cα bond length in Å.
        bond_tol:     maximum allowed deviation from ref_bond_A (default 0.15 Å).
        clash_dist:   minimum allowed non-adjacent Cα distance (default 3.5 Å).

    Returns:
        float in [0.0, 1.0] — fraction that pass both checks.
    """
    if not proposals:
        return 0.0
    n_pass = 0
    for ca in proposals:
        ca = ca.float()
        # Bond check
        bonds = (ca[1:] - ca[:-1]).norm(dim=-1)  # [N-1]
        if (bonds - ref_bond_A).abs().max().item() > bond_tol:
            continue
        # Clash check (pairwise, non-adjacent)
        N = ca.shape[0]
        if N > 1:
            diff = ca.unsqueeze(0) - ca.unsqueeze(1)  # [N, N, 3]
            dists = diff.norm(dim=-1)                   # [N, N]
            mask = torch.ones(N, N, dtype=torch.bool)
            mask.fill_diagonal_(False)
            for k in range(-1, 2):
                if k != 0:
                    idx = torch.arange(max(0, -k), min(N, N - k))
                    mask[idx, idx + k] = False
            min_noadj = dists[mask].min().item() if mask.any() else float("inf")
            if min_noadj < clash_dist:
                continue
        n_pass += 1
    return n_pass / len(proposals)


def _ca_backbone_ok(ca: torch.Tensor, max_bond_A: float = 30.0) -> bool:
    """Return True if no adjacent Cα–Cα distance exceeds max_bond_A Å.

    Catches model-collapse proposals where the diffusion model outputs Cα
    positions tens or hundreds of Å from the template.  With CA-only rigid-
    translation reconstruction (reconstruct_frame_ca), an adjacent Cα pair
    displaced by D Å produces a peptide C–N bond of ~D Å; for D ≫ 10 Å no
    amount of OpenMM minimisation can recover it.

    Args:
        ca:          [N, 3] Cα positions in Å.
        max_bond_A:  maximum allowed adjacent Cα–Cα distance (default 30 Å).
                     Round-0 proposals from an untuned universal model have bonds
                     up to ~14 Å and still sometimes succeed; round-1 collapsed
                     proposals have bonds > 30 Å and universally fail.

    Returns:
        True if max adjacent bond ≤ max_bond_A.
    """
    bonds = (ca[1:] - ca[:-1]).float().norm(dim=-1)
    return bool(bonds.max().item() <= max_bond_A)


def _min_rmsd_kabsch(query: torch.Tensor, refs: torch.Tensor) -> float:
    """Minimum Cα RMSD from query [N,3] to any frame in refs [F,N,3] via Kabsch.

    Vectorised over F — fast for up to tens of thousands of reference frames.

    Returns:
        Minimum RMSD in Å (0 if refs is empty).
    """
    if refs.shape[0] == 0:
        return 0.0
    q = query.float()                                     # [N, 3]
    r = refs.float()                                      # [F, N, 3]
    q_c = q - q.mean(0, keepdim=True)                    # center query
    r_c = r - r.mean(1, keepdim=True)                    # center each ref [F, N, 3]

    H   = torch.einsum("ni,fnj->fij", q_c, r_c)          # [F, 3, 3]
    U, _, Vt = torch.linalg.svd(H)
    d   = torch.linalg.det(Vt.mT @ U.mT)                 # [F] — reflection sign
    D   = torch.eye(3, dtype=q.dtype).unsqueeze(0).expand(H.shape[0], -1, -1).clone()
    D[:, 2, 2] = d
    R_opt  = Vt.mT @ D @ U.mT                            # [F, 3, 3]
    q_rot  = torch.einsum("fij,nj->fni", R_opt, q_c)     # [F, N, 3]
    rmsds  = (q_rot - r_c).pow(2).sum(-1).mean(-1).sqrt() # [F]
    return float(rmsds.min().item())


# ---------------------------------------------------------------------------
# bootstrap_check
# ---------------------------------------------------------------------------

def bootstrap_check(pdb_path: str, checkpoint: str, device: str,
                    bootstrap_ns: float, out_dir: str) -> dict:
    """Decide whether to start from the static PDB or run short bootstrap MD.

    Runs 20 DDIM proposals from the universal model. If geometry pass rate
    ≥ 80 %, returns a 1-frame shard from the PDB. Otherwise runs
    `bootstrap_ns` ns of OpenMM MD and returns a multi-frame shard.

    Args:
        pdb_path:     path to input heavy-atom PDB.
        checkpoint:   path to universal pretrained checkpoint (.pt).
        device:       "cuda" or "cpu".
        bootstrap_ns: MD length if bootstrap is needed (nanoseconds).
        out_dir:      directory for bootstrap MD outputs (created if needed).

    Returns:
        shard dict with keys {R, t, res_type, chain_id, res_index, n_res, dt, seq}.
    """
    from lsmd.md_validation import run_md

    shard_1f = _pdb_to_shard(pdb_path)

    # Load model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net, schedule, update_norm = load_checkpoint(ckpt, device)

    # Reference bond length from the single input frame
    ca = shard_1f["t"][0]  # [N, 3]
    ref_bond = (ca[1:] - ca[:-1]).norm(dim=-1).mean().item()

    # Generate 20 proposals (no CV guidance)
    R0 = shard_1f["R"][0].to(device)
    t0 = shard_1f["t"][0].to(device)
    proposals = []
    for _ in range(20):
        traj = rollout(
            net, schedule, update_norm, R0, t0,
            shard_1f["res_type"].to(device),
            shard_1f["chain_id"].to(device),
            shard_1f["res_index"].to(device),
            steps=50, tau_ps=2000, k=12,
            diff_steps=20, eta=1.0, temp_K=375.0,
            device=device,
        )
        proposals.append(traj[-1].cpu())

    pass_rate = _geometry_pass_rate(proposals, ref_bond_A=ref_bond)
    print(f"[bootstrap_check] geometry pass rate: {pass_rate:.1%}", flush=True)

    if pass_rate >= 0.80:
        print("[bootstrap_check] zero-MD path: universal model sufficient", flush=True)
        return shard_1f

    # Run bootstrap MD
    print(f"[bootstrap_check] pass rate < 80%; running {bootstrap_ns} ns bootstrap MD",
          flush=True)
    os.makedirs(out_dir, exist_ok=True)
    result = run_md(pdb_path, out_dir, md_ns=bootstrap_ns, temp_K=310.0)
    if result.get("error"):
        print(f"[bootstrap_check] bootstrap MD failed: {result['error']}; "
              "falling back to 1-frame shard", flush=True)
        return shard_1f

    # Load bootstrap trajectory → multi-frame shard
    traj_path = os.path.join(out_dir, "trajectory.dcd")
    top_path  = os.path.join(out_dir, "topology.pdb")
    frames    = data.load_frames(traj_path, top_path)
    # data.load_frames() returns res_type with a per-file local vocabulary.
    # Override res_type, chain_id, and res_index with canonical values from
    # _pdb_to_shard() (which uses lsmd.vocab.residue_indices()) so the model
    # always receives the same residue encoding regardless of bootstrap path.
    return {
        **frames,
        "dt":       200.0,
        "seq":      shard_1f["seq"],
        "n_res":    shard_1f["n_res"],
        "res_type": shard_1f["res_type"],
        "chain_id": shard_1f["chain_id"],
        "res_index": shard_1f["res_index"],
    }


# ---------------------------------------------------------------------------
# shard_from_md_runs
# ---------------------------------------------------------------------------

def shard_from_md_runs(md_run_dirs: list, dt_ps: float = 200.0):
    """Extract Cα backbone frames from completed OpenMM MD run directories.

    For each run directory that has a successful `metrics.json` (error == null)
    and valid `trajectory.dcd` + `topology.pdb`, loads the full-atom trajectory,
    extracts backbone SE(3) frames with `data.load_frames()`, and strides to
    approximately dt_ps ps between frames.

    Args:
        md_run_dirs: list of run directory paths (order does not matter).
        dt_ps:       desired frame spacing in ps (default 200 ps).

    Returns:
        (R, t) where R is [F_total, N, 3, 3] and t is [F_total, N, 3] float32.
        Returns (empty, empty) tensors if all runs failed or no directories given.
    """
    all_R, all_t = [], []

    for run_dir in sorted(md_run_dirs):
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as fh:
            m = json.load(fh)
        if m.get("error") is not None:
            continue

        traj_path = os.path.join(run_dir, "trajectory.dcd")
        top_path  = os.path.join(run_dir, "topology.pdb")
        if not (os.path.exists(traj_path) and os.path.exists(top_path)):
            continue

        try:
            traj = md.load(traj_path, top=top_path)
            # MDTraj's traj.timestep for OpenMM DCDs reflects the integration
            # step, not the reporter interval — use metrics.json md_ns instead.
            md_ns_run = m.get("md_ns")
            if md_ns_run and traj.n_frames > 1:
                dt_traj_ps = md_ns_run * 1000.0 / traj.n_frames
            else:
                dt_traj_ps = float(traj.timestep)
            stride = max(1, round(dt_ps / dt_traj_ps))
            traj_s = traj[::stride]

            top_obj = traj_s.topology
            n_idx  = top_obj.select("protein and name N")
            ca_idx = top_obj.select("protein and name CA")
            c_idx  = top_obj.select("protein and name C")

            xyz    = torch.tensor(traj_s.xyz, dtype=torch.float32) * 10.0  # nm → Å
            N_pos  = xyz[:, n_idx,  :]
            CA_pos = xyz[:, ca_idx, :]
            C_pos  = xyz[:, c_idx,  :]

            R_run, t_run = g.build_frames(N_pos, CA_pos, C_pos)
            all_R.append(R_run)
            all_t.append(t_run)
        except Exception as exc:
            print(f"[shard_from_md_runs] skipping {run_dir}: {exc}", flush=True)
            continue

    if not all_R:
        return torch.empty(0), torch.empty(0)

    return torch.cat(all_R, dim=0), torch.cat(all_t, dim=0)


# ---------------------------------------------------------------------------
# build_replay_shard
# ---------------------------------------------------------------------------

def build_replay_shard(new_R: torch.Tensor, new_t: torch.Tensor,
                       accumulated_pt: str, protein_meta: dict,
                       replay_cap: int = 5000, dt_ps: float = 200.0) -> dict:
    """Build a fine-tuning shard from new frames + replay of historical frames.

    Appends new_R / new_t to accumulated_pt (the growing history store), then
    returns a shard dict whose `t` and `R` are:
        all new frames  +  random_sample(history_before_this_round, n_old)
    where n_old = min(replay_cap − len(new_frames), len(history)).

    Args:
        new_R:          [F_new, N, 3, 3] rotation matrices from this round's MD.
        new_t:          [F_new, N, 3] Cα positions from this round's MD.
        accumulated_pt: path to accumulated_frames.pt (appended in-place).
        protein_meta:   dict with {res_type, chain_id, res_index, seq, n_res}.
        replay_cap:     maximum total frames in returned shard (default 5000).
        dt_ps:          frame spacing label for the shard (default 200 ps).

    Returns:
        shard dict with {res_type, chain_id, res_index, seq, n_res, R, t, dt}.
    """
    # Load existing history (frames accumulated before this round)
    if os.path.exists(accumulated_pt):
        acc = torch.load(accumulated_pt, map_location="cpu", weights_only=False)
        hist_R = acc["R"]   # [F_hist, N, 3, 3]
        hist_t = acc["t"]   # [F_hist, N, 3]
    else:
        N = new_t.shape[1]
        hist_R = torch.empty(0, N, 3, 3)
        hist_t = torch.empty(0, N, 3)

    # Append new frames to accumulated store
    updated_R = torch.cat([hist_R, new_R], dim=0)
    updated_t = torch.cat([hist_t, new_t], dim=0)
    torch.save({"R": updated_R, "t": updated_t}, accumulated_pt)

    # Build replay buffer: all new + sample of old history
    n_old = min(max(0, replay_cap - len(new_t)), len(hist_t))
    if n_old > 0 and len(hist_t) > 0:
        idx = torch.randperm(len(hist_t))[:n_old]
        combined_R = torch.cat([new_R, hist_R[idx]], dim=0)
        combined_t = torch.cat([new_t, hist_t[idx]], dim=0)
    else:
        combined_R = new_R
        combined_t = new_t

    if len(combined_t) > replay_cap:
        idx = torch.randperm(len(combined_t))[:replay_cap]
        combined_R = combined_R[idx]
        combined_t = combined_t[idx]

    return {
        **protein_meta,
        "R": combined_R,
        "t": combined_t,
        "dt": float(dt_ps),
    }


# ---------------------------------------------------------------------------
# Convergence checkers
# ---------------------------------------------------------------------------

def check_convergence(criterion: str, threshold: float, state: dict):
    """Check whether the active learning loop has converged.

    Args:
        criterion: "budget" | "coverage" | "fes"
        threshold: criterion-specific stopping value.
        state: dict with keys depending on criterion:
            budget:
                total_md_ns         (float)       — cumulative MD nanoseconds
            coverage:
                last_novel_fraction (float)       — novel fraction in last round
            fes:
                round               (int)         — current round index (0-based)
                accumulated_frames  (Tensor)      — [F, N, 3] Cα positions, all rounds
                cv_basis            (CVSpace)     — fitted collective variable space
                prev_hist           (np.ndarray | None) — [F_prev, 2] raw PC1/PC2
                                    scores from the previous round, or None if not
                                    yet available

    Returns:
        (converged: bool, metric: float)
        metric is always returned (nan when not yet computable) for logging.

    Raises:
        ValueError: if criterion is not one of the supported values.
    """
    if criterion == "budget":
        val = float(state["total_md_ns"])
        return val >= threshold, val

    elif criterion == "coverage":
        val = float(state.get("last_novel_fraction", 1.0))
        return val < threshold, val

    elif criterion == "fes":
        return _check_fes(state, threshold)

    else:
        raise ValueError(
            f"Unknown convergence criterion: {criterion!r}. "
            "Choose 'budget', 'coverage', or 'fes'."
        )


def _check_fes(state: dict, threshold: float) -> tuple:
    """JS divergence between current and previous FES histograms.

    prev_hist is expected to be a raw [F_prev, 2] numpy array of PC1/PC2
    scores from the previous round (NOT a pre-computed histogram).  Both
    rounds are histogrammed with a unified range derived from the union of
    their PC scores so that bin edges always align.

    Requires: round >= 2 AND accumulated_frames has >= 50 frames AND
    prev_hist is not None.  Returns (False, nan) whenever any condition
    is unmet.
    """
    rnd = state.get("round", 0)
    accumulated = state.get("accumulated_frames")
    prev_pc_scores = state.get("prev_hist")  # [F_prev, 2] numpy array or None
    cv_basis = state["cv_basis"]

    if rnd < 2 or accumulated is None or prev_pc_scores is None:
        return False, float("nan")
    if accumulated.shape[0] < 50:
        return False, float("nan")

    # Project current accumulated frames
    scores_curr = cv_basis.project_batch(accumulated.float())  # [F, n_cv]
    pc1_c = scores_curr[:, 0].detach().cpu().numpy()
    pc2_c = scores_curr[:, 1].detach().cpu().numpy()
    pc1_p = prev_pc_scores[:, 0]
    pc2_p = prev_pc_scores[:, 1]

    # Unified range across both rounds so histograms share bin edges
    pc1_range = [float(min(pc1_c.min(), pc1_p.min())),
                 float(max(pc1_c.max(), pc1_p.max()))]
    pc2_range = [float(min(pc2_c.min(), pc2_p.min())),
                 float(max(pc2_c.max(), pc2_p.max()))]
    # Avoid zero-width ranges
    if pc1_range[0] == pc1_range[1]:
        pc1_range[1] += 1e-6
    if pc2_range[0] == pc2_range[1]:
        pc2_range[1] += 1e-6

    bins = 50
    h_curr, _, _ = np.histogram2d(pc1_c, pc2_c, bins=bins,
                                  range=[pc1_range, pc2_range])
    h_prev, _, _ = np.histogram2d(pc1_p, pc2_p, bins=bins,
                                  range=[pc1_range, pc2_range])

    h_curr = h_curr.astype(np.float64)
    h_prev = h_prev.astype(np.float64)
    h_curr /= h_curr.sum() + 1e-10
    h_prev /= h_prev.sum() + 1e-10

    # Jensen-Shannon divergence: 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
    eps = 1e-10
    M = 0.5 * (h_curr + h_prev)
    kl_cm = float(np.sum(h_curr * np.log((h_curr + eps) / (M + eps))))
    kl_pm = float(np.sum(h_prev * np.log((h_prev + eps) / (M + eps))))
    js = 0.5 * kl_cm + 0.5 * kl_pm

    return bool(js < threshold), float(js)
