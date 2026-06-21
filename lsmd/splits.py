"""Homology-aware by-protein train/val/test splitting.

Whole CATH clusters are assigned to a single split so homologous proteins never
straddle the train/test boundary (which would inflate zero-shot scores).
"""
import random
import warnings


def by_protein_split(cluster_of, fracs=(0.8, 0.1, 0.1), seed=0):
    """Partition proteins into train/val/test by whole clusters.

    Args:
        cluster_of: {protein_id: cluster_label}.
        fracs:      (train, val, test) target fractions of proteins.
        seed:       RNG seed for deterministic cluster shuffling.

    Returns:
        {"train": [...], "val": [...], "test": [...]} sorted id lists.
    """
    clusters = {}
    for pid, cl in cluster_of.items():
        clusters.setdefault(cl, []).append(pid)
    labels = sorted(clusters)
    random.Random(seed).shuffle(labels)

    total = len(cluster_of)
    n_train = int(round(fracs[0] * total))
    n_val = int(round(fracs[1] * total))

    out = {"train": [], "val": [], "test": []}
    count = 0
    for cl in labels:
        members = clusters[cl]
        if count < n_train:
            bucket = "train"
        elif count < n_train + n_val:
            bucket = "val"
        else:
            bucket = "test"
        out[bucket].extend(members)
        count += len(members)
    result = {k: sorted(v) for k, v in out.items()}
    for split_name in ("val", "test"):
        if not result[split_name]:
            warnings.warn(
                f"by_protein_split: '{split_name}' split is empty — cluster "
                "granularity may be too coarse for the requested fracs. "
                "Try smaller fracs[0] or ensure more clusters than proteins.",
                stacklevel=2,
            )
    return result
