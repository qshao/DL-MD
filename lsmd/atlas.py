"""Per-protein preprocessing into fixed-vocabulary shards.

`build_shard` reuses the single-protein frame extraction (`data.load_frames`)
but re-keys residue identities through the global fixed vocabulary
(`vocab.residue_indices`), so residue types are comparable across proteins.
`download_atlas_entry` is a thin network wrapper around the ATLAS dataset.
`fetch_atlas_ids` returns the full list of available pdb_chain IDs from ATLAS.
"""
import io
import os
import shutil
import urllib.request
import zipfile

from lsmd import data
from lsmd import geometry as g
from lsmd import vocab

ATLAS_DT_PS = 100.0  # ATLAS analysis trajectories: 100 ns, 1 frame per 100 ps

_ATLAS_PARSABLE_URL = "https://www.dsimb.inserm.fr/ATLAS/api/parsable"
_ATLAS_PDB_FILE = "ATLAS_parsable_latest/2023_03_09_ATLAS_pdb.txt"


def fetch_atlas_ids():
    """Return list of all pdb_chain IDs available in ATLAS (e.g. '1d4t_A')."""
    with urllib.request.urlopen(_ATLAS_PARSABLE_URL, timeout=60) as resp:
        raw = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    content = zf.read(_ATLAS_PDB_FILE).decode()
    return [line.strip() for line in content.splitlines() if line.strip()]


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
    R_aa = g.so3_log(fd["R"]).half()  # [F, N, 3] axis-angle, float16
    t    = fd["t"].half()             # [F, N, 3] float16
    # Drop frames with non-finite axis-angle (degenerate near-collinear N-CA-C geometry).
    valid = R_aa.isfinite().all(dim=-1).all(dim=-1)  # [F]
    n_bad = int((~valid).sum())
    if n_bad:
        import warnings
        warnings.warn(
            f"build_shard: dropped {n_bad}/{valid.shape[0]} degenerate frames in {traj_path}"
        )
        R_aa = R_aa[valid]
        t    = t[valid]
    return {
        "R_aa": R_aa,
        "t":    t,
        "res_type": res_type,
        "chain_id": fd["chain_id"],
        "res_index": fd["res_index"],
        "dt": float(dt),
        "seq": seq,
        "n_res": len(seq),
    }


def download_atlas_entry(pdb_chain, dest_dir):
    """Download one ATLAS entry (R1 trajectory + reference PDB) into dest_dir.

    pdb_chain must be in ATLAS pdb_chain format, e.g. '1d4t_A'.
    Only the R1 replica and reference PDB are extracted; R2/R3 and TPR files
    are skipped. The downloaded ZIP is deleted after extraction.
    Returns (traj_path, top_path, dt_ps).
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{pdb_chain}"
    zip_path = os.path.join(dest_dir, f"{pdb_chain}.zip")
    traj_name = f"{pdb_chain}_R1.xtc"
    top_name  = f"{pdb_chain}.pdb"
    with urllib.request.urlopen(url, timeout=120) as resp, \
            open(zip_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(traj_name, dest_dir)
        zf.extract(top_name, dest_dir)
    os.remove(zip_path)
    return os.path.join(dest_dir, traj_name), os.path.join(dest_dir, top_name), ATLAS_DT_PS
