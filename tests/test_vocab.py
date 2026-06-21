import torch
from lsmd import vocab


def test_canonical_twenty_are_distinct():
    canon = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
             "TYR", "VAL"]
    idx = vocab.residue_indices(canon)
    assert idx.tolist() == list(range(20))
    assert vocab.N_AA_TYPES == 21


def test_aliases_map_to_canonical():
    # protonation / naming variants collapse onto their parent residue
    idx = vocab.residue_indices(["HIE", "HID", "HIP", "CYX"])
    his = vocab.residue_indices(["HIS"])[0].item()
    cys = vocab.residue_indices(["CYS"])[0].item()
    assert idx.tolist() == [his, his, his, cys]


def test_unknown_maps_to_unk_and_is_case_insensitive():
    idx = vocab.residue_indices(["XYZ", "ala"])
    assert idx[0].item() == vocab.UNK_INDEX == 20
    assert idx[1].item() == vocab.residue_indices(["ALA"])[0].item()
    assert idx.dtype == torch.long
