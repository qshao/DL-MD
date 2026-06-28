# Tau K18 Active Learning — Test Run Findings

**Structure:** PDB 6QJH chain A (Tau 272–330, 59 residues, cryo-EM filament core)  
**Run:** 3 rounds, 30 proposals/round, 5 MD×0.5 ns/round, 500 fine-tune steps

## What worked

- Loop ran end-to-end (bootstrap → proposals → MD → fine-tune → .done) correctly.
- 100% novelty every round — the universal model consistently explores conformations
  outside the single-frame seed, even for an unseen IDP.
- Bootstrap MD triggered correctly (geometry pass rate 0%, expected for an IDP).

## Issues found during testing

### 1. C-terminal HIS missing OXT (fixed in `lsmd/md_validation.py`)

Crystal/cryo-EM PDBs loaded via mdtraj lack the OXT atom on the C-terminal
residue. OpenMM's `addHydrogens()` fails with "No template found for residue
N (HIS) — missing 1 C atom."

**Fix:** `_prepare_pdb()` in `md_validation.py` now adds OXT geometrically and
renames HIS→HIE (unambiguous protonation) before OpenMM sees the file.

### 2. Reconstructed IDP structures unfold immediately

Round 1–2 MD RMSD reached 20–27 Å at end of 0.5 ns. The all-atom reconstruction
from a 1-Cα seed onto a disordered conformation produces heavy-atom clashes that
the energy minimiser cannot fully resolve in 5000 steps, causing NaN blowups or
complete unfolding.

### 3. PDB serial number overflow (rare, not yet fixed)

One structure in round 2 failed with `"could not convert string to float: '5 894.84'"`,
indicating that after hydrogen addition the PDB written by OpenMM had a corrupted
column (serial > 99999 overflowed the 5-char PDB field, shifting coordinates).

## Recommendations for a production Tau run

```bash
python scripts/active_learning.py \
    --pdb              tau_test/tau_K18_chainA.pdb \
    --checkpoint       checkpoints/v2_256h_90k.pt \
    --out              tau_K18_production \
    --rounds           20 \
    --proposals        100 \
    --batch-size       10 \
    --md-ns            2 \          # longer → more frames per run, fewer blow-ups
    --bootstrap-ns     5 \          # more bootstrap data for initial CV fitting
    --replay-cap       5000 \
    --novel-threshold  1.5 \
    --stop             coverage \
    --stop-threshold   0.10 \
    --fine-tune-steps  2000 \       # more steps once replay buffer is non-trivial
    --n-parallel       4 \
    --device           cuda
```

Key changes vs test run:
- `--md-ns 2`: longer trajectories give the energy minimiser more time to relax
  clashes before the production run, reducing NaN blow-ups.
- `--fine-tune-steps 2000`: 500 steps overfits a 2–7 frame buffer trivially
  (loss→0 in <100 steps); 2000 steps is more appropriate as frames accumulate.
- `--stop coverage 0.10`: more meaningful than budget for an IDP — stop when
  fewer than 10% of proposals are novel (landscape well-sampled).
- Prepare the input PDB with pdbfixer first if available:
  ```python
  from pdbfixer import PDBFixer
  from openmm.app import PDBFile
  fixer = PDBFixer(filename="tau_K18_chainA.pdb")
  fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
  fixer.addMissingHydrogens(7.0)
  with open("tau_K18_ready.pdb", "w") as f:
      PDBFile.writeFile(fixer.topology, fixer.positions, f)
  ```
  This adds OXT, caps termini, and assigns protonation states before the loop
  starts — making `_prepare_pdb()` a no-op fallback rather than the primary path.
