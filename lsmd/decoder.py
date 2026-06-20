import torch
from lsmd import geometry as g
from lsmd import featurize as f

PEPTIDE_CN = 1.33  # Angstrom
CLASH = 2.0        # min non-bonded CA-CA-ish distance for penalty


def decode_frames(R_t, t_t, u_samples):
    """Decode updates into full frame trajectories.

    Args:
        R_t: Target rotation [N, 3, 3]
        t_t: Target translation [N, 3]
        u_samples: Update samples [K, N, 6]

    Returns:
        (R_f, t_f): Frame rotations [K, N, 3, 3] and translations [K, N, 3]
    """
    K = u_samples.shape[0]
    # Broadcast R_t, t_t over K dim and apply all updates in one vectorised call
    R_t_b = R_t.unsqueeze(0).expand(K, -1, -1, -1)   # [K, N, 3, 3]
    t_t_b = t_t.unsqueeze(0).expand(K, -1, -1)        # [K, N, 3]
    return f.apply_update(R_t_b, t_t_b, u_samples)    # [K, N, 3, 3], [K, N, 3]


def build_structure(R, t):
    """Build atom coordinates from frames.

    Args:
        R: Rotations [N, 3, 3]
        t: Translations [N, 3]

    Returns:
        atoms: Atom coordinates [N, 4, 3] (N, CA, C, O)
    """
    return g.place_backbone(R, t)  # [N,4,3]


def peptide_bond_violation(atoms):
    """Compute mean deviation of peptide C-N bond length from ideal.

    Args:
        atoms: Atom coordinates [N, 4, 3]

    Returns:
        scalar: Mean absolute deviation from PEPTIDE_CN (1.33 Angstrom)
    """
    C = atoms[:-1, 2, :]   # C of residue i
    N = atoms[1:, 0, :]    # N of residue i+1
    d = (C - N).norm(dim=-1)
    return (d - PEPTIDE_CN).abs().mean()


def _clash_penalty(ca):
    """Compute penalty for clashes between non-bonded CA atoms.

    Args:
        ca: CA coordinates [N, 3]

    Returns:
        scalar: Sum of squared clash violations
    """
    d = torch.cdist(ca, ca)
    n = ca.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    # only penalize non-adjacent residues
    for i in range(n - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False
    viol = (CLASH - d).clamp_min(0.0) * mask
    return (viol ** 2).sum()


def idealize(atoms, steps=50):
    """Optimize per-residue translation to close peptide bonds and remove clashes.

    Args:
        atoms: Atom coordinates [N, 4, 3]
        steps: Number of optimization steps

    Returns:
        atoms: Idealized atom coordinates [N, 4, 3]

    Strategy: Optimize a per-residue translation delta [N, 3] that shifts each
    residue rigidly (keeping internal geometry fixed) to minimize peptide bond
    violations and clashes. Uses Adam with lr=0.05.
    """
    # optimize a per-residue translation that closes peptide bonds and removes clashes,
    # keeping each residue rigid (only CA position shifts; relative atom geometry preserved)
    delta = torch.zeros(atoms.shape[0], 3, requires_grad=True, device=atoms.device)
    opt = torch.optim.Adam([delta], lr=0.05)
    base = atoms.detach()
    for _ in range(steps):
        opt.zero_grad()
        shifted = base + delta.unsqueeze(1)
        l_pep = ((shifted[:-1, 2, :] - shifted[1:, 0, :]).norm(dim=-1) - PEPTIDE_CN).pow(2).sum()
        l_clash = _clash_penalty(shifted[:, 1, :])
        loss = l_pep + 0.1 * l_clash
        loss.backward()
        opt.step()
    return (base + delta.detach().unsqueeze(1))


_ELEMENTS = {"N": "N", "CA": "C", "C": "C", "O": "O"}
_ATOM_NAMES = ["N", "CA", "C", "O"]


def write_pdb(atoms, res_type_names, path):
    """Write structure to PDB file.

    Args:
        atoms: Atom coordinates [N, 4, 3]
        res_type_names: List of residue type names (length N), e.g. ["ALA", "GLY", ...]
        path: Output file path

    Returns:
        None (writes to file)

    PDB format: standard ATOM records with serial counter, atom name,
    residue type, chain A, 1-based residue number, xyz coordinates.
    """
    lines = []
    serial = 1
    for ri in range(atoms.shape[0]):
        for ai, name in enumerate(_ATOM_NAMES):
            x, y, z = atoms[ri, ai].tolist()
            lines.append(
                f"ATOM  {serial:5d} {name:<4s}{res_type_names[ri]:>3s} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {_ELEMENTS[name]:>2s}"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


_2BEAD_ATOM_NAMES = [" CA ", " CB "]
_2BEAD_ELEMENTS   = ["C",    "C"  ]

_4BEAD_ATOM_NAMES = [" N  ", " CA ", " C  ", " CB "]
_4BEAD_ELEMENTS   = ["N",    "C",    "C",    "C"  ]


def write_2bead_pdb(beads, res_type_names, path, gly_mask=None):
    """Write a 2-bead (CA, CB) trace to a PDB file.

    Args:
        beads:          [P, 2, 3] bead coordinates in Å, order (CA, CB)
        res_type_names: list of P residue name strings
        path:           output file path
        gly_mask:       optional bool tensor [P]; if True, CB is omitted (Glycine)
    """
    lines  = []
    serial = 1
    for ri in range(beads.shape[0]):
        for ai, (aname, elem) in enumerate(zip(_2BEAD_ATOM_NAMES, _2BEAD_ELEMENTS)):
            if ai == 1 and gly_mask is not None and gly_mask[ri]:
                continue
            x, y, z = beads[ri, ai].tolist()
            lines.append(
                f"ATOM  {serial:5d} {aname} {res_type_names[ri]:>3s} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_4bead_pdb(beads, res_type_names, path, gly_mask=None):
    """Write a 4-bead CA trace (N, CA, C, CB) to a PDB file.

    Args:
        beads:          [P, 4, 3] bead coordinates in Å, order (N, CA, C, CB)
        res_type_names: list of P residue name strings, e.g. ["ALA", "GLY", ...]
        path:           output file path
        gly_mask:       optional bool tensor [P]; if True for residue i, the CB
                        atom is omitted (Glycine has no real CB)
    """
    lines  = []
    serial = 1
    for ri in range(beads.shape[0]):
        for ai, (aname, elem) in enumerate(zip(_4BEAD_ATOM_NAMES, _4BEAD_ELEMENTS)):
            if ai == 3 and gly_mask is not None and gly_mask[ri]:
                continue   # skip placeholder CB for Gly
            x, y, z = beads[ri, ai].tolist()
            lines.append(
                f"ATOM  {serial:5d} {aname} {res_type_names[ri]:>3s} A{ri + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_ca_pdb(ca, res_type_names, path):
    """Write a CA-only trace to a PDB file (one CA atom per residue).

    Args:
        ca: CA coordinates [P, 3]
        res_type_names: list of residue names (length P), e.g. ["ALA", ...]
        path: output file path

    Returns:
        None (writes to file).
    """
    lines = []
    for ri in range(ca.shape[0]):
        x, y, z = ca[ri].tolist()
        lines.append(
            f"ATOM  {ri + 1:5d}  CA  {res_type_names[ri]:>3s} A{ri + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
