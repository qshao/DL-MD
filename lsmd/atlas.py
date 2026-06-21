"""Per-protein preprocessing into fixed-vocabulary shards.

`build_shard` reuses the single-protein frame extraction (`data.load_frames`)
but re-keys residue identities through the global fixed vocabulary
(`vocab.residue_indices`), so residue types are comparable across proteins.
`download_atlas_entry` is a thin network wrapper around the ATLAS dataset.
"""
import os
from lsmd import data
from lsmd import vocab


def build_shard(traj_path, top_path, dt):
    """Build one fixed-vocab shard from a trajectory + topology.

    Args:
        traj_path: trajectory file path.
        top_path:  topology file path.
        dt:        picoseconds per frame.

    Returns:
        dict with R [F,N,3,3], t [F,N,3], res_type [N] (fixed vocab 0..20),
        chain_id [N], res_index [N], dt (float), seq (list[str]), n_res (int).
    """
    fd = data.load_frames(traj_path, top_path)
    seq = list(fd["res_names"])
    res_type = vocab.residue_indices(seq)
    return {
        "R": fd["R"],
        "t": fd["t"],
        "res_type": res_type,
        "chain_id": fd["chain_id"],
        "res_index": fd["res_index"],
        "dt": float(dt),
        "seq": seq,
        "n_res": len(seq),
    }


def download_atlas_entry(pdbid, dest_dir):
    """Download one ATLAS entry (trajectory + reference) into dest_dir.

    Thin wrapper around the ATLAS analysis-trajectory download. Network I/O;
    not unit-tested. Returns (traj_path, top_path, dt_ps).

    ATLAS analysis trajectories are saved at 10 ps/frame; adjust if the chosen
    ATLAS product differs.
    """
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis"
    url = f"{base}/{pdbid}/{pdbid}_analysis.zip"
    zip_path = os.path.join(dest_dir, f"{pdbid}.zip")
    urllib.request.urlretrieve(url, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    traj_path = os.path.join(dest_dir, f"{pdbid}_prod_R1_fit.xtc")
    top_path = os.path.join(dest_dir, f"{pdbid}.pdb")
    return traj_path, top_path, 10.0
