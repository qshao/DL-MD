import os
import torch
import pytest
from lsmd import transfer_train as tt
from lsmd import transfer_eval as te
from lsmd import splits
from lsmd import geometry as g


def _synthetic_shard(F=20, N=10, dt=200.0, seed=0):
    torch.manual_seed(seed)
    return {
        "R": g.so3_exp(torch.randn(F, N, 3) * 0.1),
        "t": torch.randn(F, N, 3) * 5.0,
        "res_type": torch.randint(0, 21, (N,)),
        "chain_id": torch.zeros(N, dtype=torch.long),
        "res_index": torch.arange(N),
        "dt": dt, "seq": ["ALA"] * N, "n_res": N,
    }


def test_baseline_pipeline_composes_on_synthetic():
    cluster_of = {f"p{i}": f"c{i // 2}" for i in range(6)}
    # 3 clusters of 2; balanced thirds give 1 cluster per split
    sp = splits.by_protein_split(cluster_of, fracs=(0.34, 0.33, 0.33), seed=0)
    assert sp["train"] and sp["test"]

    shards = {f"p{i}": _synthetic_shard(N=10, seed=i) for i in range(6)}
    train_shards = [shards[i] for i in sp["train"]]
    ckpt = tt.train(train_shards, lags_ps=[200.0], k=4, hidden=16, layers=2,
                    max_union_nodes=25, accum=2, steps=3, T_diff=20,
                    norm_samples=16, device="cpu", seed=0)

    held = shards[sp["test"][0]]
    net, sched, norm = te.load_checkpoint(ckpt, device="cpu")
    traj = te.rollout(net, sched, norm, held["R"][0], held["t"][0],
                      held["res_type"], held["chain_id"], held["res_index"],
                      steps=4, tau_ps=200.0, k=4, diff_steps=3, device="cpu")
    metrics = te.evaluate(traj, held["t"])
    assert -1.0 <= metrics["rmsf_corr"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["dist_js"]))


def test_real_wt_shard_feeds_trainer_if_available():
    trr, gro = "WT/WT-sol6.trr", "WT/WT-sol6.gro"
    if not (os.path.exists(trr) and os.path.exists(gro)):
        pytest.skip("WT trajectory not present")
    from lsmd import atlas
    shard = atlas.build_shard(trr, gro, dt=200.0)
    ckpt = tt.train([shard], lags_ps=[200.0, 1000.0], k=8, hidden=16, layers=2,
                    max_union_nodes=10_000, accum=1, steps=2, T_diff=20,
                    norm_samples=8, device="cpu", seed=0)
    assert ckpt["n_aa_types"] == 21
