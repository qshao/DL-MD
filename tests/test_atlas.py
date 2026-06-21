import os
import torch
import pytest
from lsmd import atlas, vocab


def test_load_frames_exposes_res_names():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    from lsmd import data
    fd = data.load_frames(trr, gro)
    assert "res_names" in fd
    assert len(fd["res_names"]) == fd["res_type"].shape[0]


def test_build_shard_uses_fixed_vocab_and_keys():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    shard = atlas.build_shard(trr, gro, dt=200.0)
    N = shard["n_res"]
    assert shard["R"].shape[1] == N and shard["t"].shape[1] == N
    assert shard["res_type"].shape == (N,)
    # fixed vocab range
    assert int(shard["res_type"].min()) >= 0
    assert int(shard["res_type"].max()) <= vocab.N_AA_TYPES - 1
    # res_type must equal the vocab keying of the stored sequence
    assert torch.equal(shard["res_type"], vocab.residue_indices(shard["seq"]))
    assert shard["dt"] == 200.0
    assert len(shard["seq"]) == N
