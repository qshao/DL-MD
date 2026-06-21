"""Fixed amino-acid vocabulary shared across all proteins.

Identical indexing for every protein so residue identity is comparable
across the training corpus. Index 20 is the UNK catch-all.
"""
import torch

CANONICAL = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
             "TYR", "VAL"]

UNK_INDEX = len(CANONICAL)          # 20
N_AA_TYPES = len(CANONICAL) + 1     # 21 (20 canonical + UNK)

# Common protonation / naming variants → canonical parent.
_ALIASES = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "ARN": "ARG", "MSE": "MET",
}

_INDEX = {name: i for i, name in enumerate(CANONICAL)}


def residue_indices(res_names):
    """Map 3-letter residue names to fixed vocabulary indices.

    Args:
        res_names: iterable of residue name strings (any case).

    Returns:
        LongTensor [len(res_names)] with values in 0..20; unknown → UNK_INDEX.
    """
    out = []
    for nm in res_names:
        key = str(nm).strip().upper()
        key = _ALIASES.get(key, key)
        out.append(_INDEX.get(key, UNK_INDEX))
    return torch.tensor(out, dtype=torch.long)
