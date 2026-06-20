import torch
from lsmd import geometry as g
from lsmd import featurize as f
from lsmd import decoder as dec


def test_decode_frames_roundtrip():
    R_t = g.so3_exp(torch.randn(5, 3) * 0.3)
    t_t = torch.randn(5, 3)
    R_f = g.so3_exp(torch.randn(5, 3) * 0.3)
    t_f = torch.randn(5, 3)
    u = f.relative_update(R_t, t_t, R_f, t_f).unsqueeze(0)  # [1,5,6]
    R_d, t_d = dec.decode_frames(R_t, t_t, u)
    assert torch.allclose(R_d[0], R_f, atol=1e-4)
    assert torch.allclose(t_d[0], t_f, atol=1e-4)


def test_build_structure_shape():
    R = g.so3_exp(torch.randn(6, 3) * 0.2)
    t = torch.randn(6, 3)
    atoms = dec.build_structure(R, t)
    assert atoms.shape == (6, 4, 3)


def test_idealize_reduces_peptide_violation():
    # lay residues far apart so peptide bonds are badly broken
    R = g.so3_exp(torch.zeros(5, 3))
    t = torch.arange(5).float().unsqueeze(-1).repeat(1, 3) * 5.0
    atoms = dec.build_structure(R, t)
    before = dec.peptide_bond_violation(atoms)
    fixed = dec.idealize(atoms, steps=200)
    after = dec.peptide_bond_violation(fixed)
    assert after < before


def test_write_pdb(tmp_path):
    R = g.so3_exp(torch.randn(3, 3) * 0.2)
    t = torch.randn(3, 3)
    atoms = dec.build_structure(R, t)
    p = tmp_path / "out.pdb"
    dec.write_pdb(atoms, ["ALA", "GLY", "ALA"], str(p))
    text = p.read_text()
    assert "ATOM" in text and "CA" in text
