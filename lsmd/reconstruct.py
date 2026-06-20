"""All-atom reconstruction from coarse-grained bead trajectories.

Strategy
--------
For every generated frame the nearest frame in the original MD trajectory is
found by CA-RMSD.  That frame provides realistic sidechain conformations.

4-bead (N, CA, C, CB)
    Per-residue Kabsch superposition of the template backbone (N, CA, C) onto
    the generated backbone places the deep sidechain atoms (CG, CD, …) in the
    correct local frame.  N, CA, C, CB are then set directly from the generated
    coordinates; the carbonyl O is placed from the peptide-plane geometry.

2-bead (CA, CB) and CA-only
    Each residue is translated rigidly by (CA_gen − CA_template), preserving the
    template rotamer in the global frame.

Output: protein heavy atoms only (no H, no solvent).
"""

import numpy as np
import torch
import mdtraj as md


_C_O_BOND = 1.229   # Å — ideal carbonyl C=O bond length


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def place_oxygen(N, CA, C):
    """Place carbonyl O for each residue using peptide-plane geometry.

    For residues 0..P-2 the O is placed in the plane of CA(i)–C(i)–N(i+1),
    anti to N(i+1): C–O = 1.229 Å.  For the C-terminal residue the O is
    placed along the C–CA extension.

    Args:
        N:  [P, 3] nitrogen positions
        CA: [P, 3] alpha-carbon positions
        C:  [P, 3] carbonyl-carbon positions

    Returns:
        O: [P, 3] carbonyl oxygen positions
    """
    P = N.shape[0]
    O = torch.zeros(P, 3, dtype=C.dtype, device=C.device)

    C_i   = C[:-1]
    CA_i  = CA[:-1]
    N_ip1 = N[1:]

    u_CCA = CA_i - C_i
    u_CCA = u_CCA / u_CCA.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u_CN  = N_ip1 - C_i
    u_CN  = u_CN  / u_CN.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # O bisects the exterior of the CA-C-N angle (trans to both CA and N)
    o_dir = -(u_CCA + u_CN)
    o_dir = o_dir / o_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    O[:-1] = C_i + _C_O_BOND * o_dir

    # C-terminal: extend along C–CA axis
    last_dir = C[-1] - CA[-1]
    last_dir = last_dir / last_dir.norm().clamp_min(1e-8)
    O[-1] = C[-1] + _C_O_BOND * last_dir

    return O


def _kabsch_rt(src, tgt):
    """SVD Kabsch: (R, t) such that R @ src[i] + t ≈ tgt[i] for [3, 3] inputs."""
    s_mean = src.mean(0)
    t_mean = tgt.mean(0)
    H = (src - s_mean).T @ (tgt - t_mean)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=src.dtype, device=src.device))
    R = Vt.T @ D @ U.T
    t = t_mean - R @ s_mean
    return R, t


# ---------------------------------------------------------------------------
# Reconstructor class
# ---------------------------------------------------------------------------

class AllAtomReconstructor:
    """Reconstruct all-atom heavy-atom structures from bead-model trajectories.

    Typical use::

        rec = AllAtomReconstructor("WT/WT-sol6.trr", "WT/WT-sol6.gro")
        traj_out = rec.reconstruct_trajectory(bead_frames, mode="4bead",
                                               gly_mask=gly_mask)
        traj_out.save_pdb("allatom_gen.pdb")
    """

    def __init__(self, traj_path, top_path):
        """Load and cache the MD template trajectory.

        Args:
            traj_path: trajectory file (TRR, DCD, XTC, …)
            top_path:  topology file (GRO, PDB, …)
        """
        print(f"Loading template trajectory {traj_path} …", end=" ", flush=True)
        traj = md.load(traj_path, top=top_path)
        if traj.unitcell_lengths is not None:
            traj.make_molecules_whole(inplace=True)
        top = traj.topology
        ca_all = top.select("protein and name CA")
        if len(ca_all) > 0:
            traj.superpose(traj, 0, atom_indices=ca_all)
        print(f"{traj.n_frames} frames, {traj.n_atoms} atoms")

        def _aidx(res, name):
            for a in res.atoms:
                if a.name == name:
                    return a.index
            return None

        # Select protein residues with complete backbone
        res_list = []
        for r in top.residues:
            if r.name == "HOH":
                continue
            if any(_aidx(r, nm) is None for nm in ("N", "CA", "C")):
                continue
            res_list.append(r)

        P = len(res_list)
        self.P = P
        print(f"  {P} reconstructable residues")

        # Per-residue atom index tables
        n_idx, ca_idx, c_idx, o_idx, cb_idx = [], [], [], [], []
        sc_idx = []          # heavy sidechain atoms beyond CB (CG, CD, …)
        all_heavy_idx = []   # all heavy atoms in residue

        _bb = {"N", "CA", "C", "O", "CB"}

        for r in res_list:
            ai = {a.name: a.index for a in r.atoms}
            n_idx.append(ai["N"])
            ca_idx.append(ai["CA"])
            c_idx.append(ai["C"])
            o_idx.append(ai.get("O"))
            cb_idx.append(ai.get("CB"))

            sc = [idx for nm, idx in ai.items()
                  if nm not in _bb and top.atom(idx).element.symbol != "H"]
            sc_idx.append(sc)

            heavy = [idx for nm, idx in ai.items()
                     if top.atom(idx).element.symbol != "H"]
            all_heavy_idx.append(heavy)

        self._n_idx  = n_idx
        self._ca_idx = ca_idx
        self._c_idx  = c_idx
        self._o_idx  = o_idx
        self._cb_idx = cb_idx
        self._sc_idx = sc_idx
        self._all_heavy_idx = all_heavy_idx

        # Output topology: protein heavy atoms only
        prot_ha = sorted({idx for per_res in all_heavy_idx for idx in per_res})
        self._prot_ha = np.array(prot_ha, dtype=int)
        self._out_top = traj.atom_slice(self._prot_ha).topology

        # Cache full xyz in Å  [F, N_atoms, 3]
        xyz_nm = np.array(traj.xyz)
        self._xyz_A = torch.tensor(xyz_nm * 10.0, dtype=torch.float32)

        # Cache backbone positions for fast nearest-frame lookup  [F, P, 3]
        self._ca_tmpl = self._xyz_A[:, ca_idx, :]
        self._n_tmpl  = self._xyz_A[:, n_idx,  :]
        self._c_tmpl  = self._xyz_A[:, c_idx,  :]

    # ------------------------------------------------------------------

    def _nearest_frame(self, ca_gen):
        """Index of the template frame closest to ca_gen [P, 3] by CA-RMSD."""
        ca_q = ca_gen - ca_gen.mean(0)
        ca_t = self._ca_tmpl - self._ca_tmpl.mean(1, keepdim=True)
        sq = ((ca_t - ca_q.unsqueeze(0)) ** 2).sum(-1).mean(-1)
        return int(sq.argmin().item())

    # ------------------------------------------------------------------

    def reconstruct_frame_4bead(self, beads, gly_mask=None):
        """All-atom heavy-atom frame from [P, 4, 3] bead coordinates.

        For each residue:
          1. Kabsch-align template (N, CA, C) → generated (N, CA, C)
          2. Transform deep sidechain atoms (CG, CD, …) by that rotation
          3. Set N, CA, C from generated; O from peptide geometry; CB from
             generated (non-Gly) or transformed template CB (Gly)

        Args:
            beads:    [P, 4, 3] tensor, atom order (N, CA, C, CB)
            gly_mask: [P] bool — True marks Gly residues (CB is a placeholder)

        Returns:
            xyz: [N_heavy, 3] float32 numpy array in Å
        """
        beads = beads.float()
        N_gen  = beads[:, 0, :]
        CA_gen = beads[:, 1, :]
        C_gen  = beads[:, 2, :]
        CB_gen = beads[:, 3, :]

        fi  = self._nearest_frame(CA_gen)
        xyz = self._xyz_A[fi].clone()   # [N_total_atoms, 3]

        O_gen  = place_oxygen(N_gen, CA_gen, C_gen)
        N_t    = self._n_tmpl[fi]
        CA_t   = self._ca_tmpl[fi]
        C_t    = self._c_tmpl[fi]

        for i in range(self.P):
            src = torch.stack([N_t[i], CA_t[i], C_t[i]])
            tgt = torch.stack([N_gen[i], CA_gen[i], C_gen[i]])
            R, t = _kabsch_rt(src, tgt)

            for ai in self._sc_idx[i]:
                xyz[ai] = R @ xyz[ai] + t

            # Place backbone from generated / geometry
            xyz[self._n_idx[i]]  = N_gen[i]
            xyz[self._ca_idx[i]] = CA_gen[i]
            xyz[self._c_idx[i]]  = C_gen[i]
            if self._o_idx[i] is not None:
                xyz[self._o_idx[i]] = O_gen[i]

            # CB: generated for non-Gly; Kabsch-transformed for Gly
            if self._cb_idx[i] is not None:
                is_gly = gly_mask is not None and bool(gly_mask[i])
                if is_gly:
                    xyz[self._cb_idx[i]] = R @ xyz[self._cb_idx[i]] + t
                else:
                    xyz[self._cb_idx[i]] = CB_gen[i]

        return xyz[self._prot_ha].numpy()

    def reconstruct_frame_ca(self, ca_gen):
        """All-atom heavy-atom frame from [P, 3] CA positions (CA or 2-bead).

        Finds the nearest template frame and translates each residue rigidly
        by (CA_gen − CA_template), preserving the template rotamer.

        Args:
            ca_gen: [P, 3] generated CA positions

        Returns:
            xyz: [N_heavy, 3] float32 numpy array in Å
        """
        ca_gen = ca_gen.float()
        fi  = self._nearest_frame(ca_gen)
        xyz = self._xyz_A[fi].clone()

        delta = ca_gen - self._ca_tmpl[fi]   # per-residue CA displacement [P, 3]
        for i in range(self.P):
            shift = delta[i]
            for ai in self._all_heavy_idx[i]:
                xyz[ai] = xyz[ai] + shift

        return xyz[self._prot_ha].numpy()

    # ------------------------------------------------------------------

    def reconstruct_trajectory(self, bead_frames, mode="4bead", gly_mask=None):
        """Reconstruct a full trajectory from generated bead frames.

        Args:
            bead_frames: [F, P, n_beads, 3]  (n_beads=4/2/1) or [F, P, 3] for CA
            mode:        "4bead" | "2bead" | "ca"
            gly_mask:    [P] bool — Gly residues (4-bead only)

        Returns:
            mdtraj.Trajectory with F frames, protein heavy atoms only
        """
        F = bead_frames.shape[0]
        xyz_list = []

        for fi in range(F):
            frame = bead_frames[fi]
            if mode == "4bead":
                xyz = self.reconstruct_frame_4bead(frame, gly_mask=gly_mask)
            elif mode == "2bead":
                xyz = self.reconstruct_frame_ca(frame[:, 0, :])   # CA at index 0
            else:
                # CA-only: frame is [P, 3]
                f2 = frame if frame.ndim == 2 else frame[:, 0, :]
                xyz = self.reconstruct_frame_ca(f2)
            xyz_list.append(xyz)
            if (fi + 1) % 10 == 0 or fi == F - 1:
                print(f"  Reconstructed {fi + 1}/{F} frames", end="\r")

        print()
        xyz_nm = np.stack(xyz_list, axis=0) / 10.0   # Å → nm for MDtraj
        return md.Trajectory(xyz_nm, self._out_top)
