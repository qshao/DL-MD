from lsmd import splits


def _toy():
    # 10 proteins in 5 clusters (2 each)
    cluster_of = {}
    for c in range(5):
        cluster_of[f"p{2*c}"] = f"cath{c}"
        cluster_of[f"p{2*c+1}"] = f"cath{c}"
    return cluster_of


def test_splits_are_disjoint_and_cover_all():
    s = splits.by_protein_split(_toy(), fracs=(0.6, 0.2, 0.2), seed=0)
    all_ids = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert all_ids == set(_toy().keys())
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["val"]) & set(s["test"]))


def test_no_cluster_spans_two_splits():
    cluster_of = _toy()
    s = splits.by_protein_split(cluster_of, fracs=(0.6, 0.2, 0.2), seed=1)
    where = {}
    for name, ids in s.items():
        for pid in ids:
            where[cluster_of[pid]] = where.get(cluster_of[pid], set()) | {name}
    assert all(len(v) == 1 for v in where.values())


def test_deterministic_for_fixed_seed():
    a = splits.by_protein_split(_toy(), seed=3)
    b = splits.by_protein_split(_toy(), seed=3)
    assert a == b
