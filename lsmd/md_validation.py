"""OpenMM implicit-solvent MD validation for all-atom protein structures."""
import json
import os
import tempfile

try:
    import openmm as omm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def _prepare_pdb(pdb_path: str) -> str:
    """Return a path to a PDB that OpenMM's amber14 force field can parse.

    Handles two common issues in crystal/cryo-EM structures:
      1. HIS residues with ambiguous protonation — renamed to HIE (Ne-protonated).
      2. C-terminal residue missing OXT — computed from C/O/CA geometry.

    Returns the original path when neither fix is needed (avoids temp file I/O).
    """
    import numpy as np

    with open(pdb_path) as fh:
        lines = fh.readlines()

    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        return pdb_path

    # ── Collect last-residue info BEFORE any renaming ────────────────────────
    last_chain  = atom_lines[-1][21]
    last_res_id = atom_lines[-1][22:26].strip()
    last_atoms  = {
        l[12:16].strip(): l for l in atom_lines
        if l[22:26].strip() == last_res_id and l[21] == last_chain
    }
    needs_oxt = "OXT" not in last_atoms and "OT2" not in last_atoms
    has_his   = any(l[17:20] == "HIS" for l in atom_lines)

    if not needs_oxt and not has_his:
        return pdb_path

    # ── Rename HIS → HIE in all ATOM/HETATM and TER lines ──────────────────
    new_lines = []
    for l in lines:
        if (l.startswith(("ATOM", "HETATM", "TER"))) and len(l) > 20 and l[17:20] == "HIS":
            l = l[:17] + "HIE" + l[20:]
        new_lines.append(l)

    # ── Add OXT to C-terminal residue if missing ─────────────────────────────
    if needs_oxt:
        def _xyz(raw_line):
            return np.array([float(raw_line[30:38]),
                             float(raw_line[38:46]),
                             float(raw_line[46:54])])

        C_line  = last_atoms.get("C")
        O_line  = last_atoms.get("O")
        CA_line = last_atoms.get("CA")
        if C_line and O_line and CA_line:
            C, O, CA = _xyz(C_line), _xyz(O_line), _xyz(CA_line)
            # Place OXT as the carboxylate mirror of O through the C-CA bond axis
            u   = (C - CA); u /= np.linalg.norm(u)
            CO  = O - C
            OXT = C + (CO - 2.0 * np.dot(CO, u) * u)

            # Use the renamed residue name (HIE if originally HIS, else original)
            res_name = "HIE" if last_atoms["C"][17:20] == "HIS" else last_atoms["C"][17:20]

            # Serial: one past the last ATOM/HETATM in new_lines
            last_ser = max(
                int(l[6:11]) for l in new_lines if l.startswith(("ATOM", "HETATM"))
            )
            oxt_line = (
                f"ATOM  {last_ser+1:5d}  OXT {res_name} {last_chain}"
                f"{last_res_id.rjust(4)}    "
                f"{OXT[0]:8.3f}{OXT[1]:8.3f}{OXT[2]:8.3f}"
                f"  1.00  0.00           O  \n"
            )
            insert_idx = next(
                (i for i, l in enumerate(new_lines)
                 if l.startswith("TER") or l.startswith("END")),
                len(new_lines),
            )
            new_lines.insert(insert_idx, oxt_line)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False)
    tmp.writelines(new_lines)
    tmp.flush()
    return tmp.name


def run_md(pdb_path, out_dir, md_ns, temp_K=310.0, n_steps_min=5000):
    """Run AMBER14/GBn2 implicit-solvent MD on a heavy-atom PDB structure.

    Args:
        pdb_path (str):   Path to all-atom heavy-atom PDB (no H, no solvent).
        out_dir (str):    Directory for trajectory.dcd, topology.pdb, metrics.json.
        md_ns (float):    Simulation length in nanoseconds.
        temp_K (float):   Temperature in Kelvin (default 310.0).
        n_steps_min (int):Energy minimisation steps (default 1000).

    Returns:
        dict: id, md_ns, final_pe_kJ, rmsd_initial_A, rmsd_final_A,
              rmsd_mean_A, rmsd_std_A, stable, error
    """
    struct_id = os.path.splitext(os.path.basename(pdb_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")

    # Checkpoint: return cached result only when the previous run succeeded.
    # A cached error (GPU OOM, bad PDB, etc.) should NOT be treated as permanent —
    # fall through and re-run so transient failures can be recovered.
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            cached = json.load(fh)
        if cached.get("error") is None:
            return cached  # only skip on successful previous run
        # else fall through and re-run

    if not HAS_OPENMM:
        raise ImportError(
            "openmm is required: conda install -c conda-forge openmm"
        )

    traj_path = os.path.join(out_dir, "trajectory.dcd")
    top_path  = os.path.join(out_dir, "topology.pdb")

    result = {
        "id": struct_id, "md_ns": md_ns,
        "final_pe_kJ": None,
        "rmsd_initial_A": None, "rmsd_final_A": None,
        "rmsd_mean_A": None, "rmsd_std_A": None,
        "stable": False, "error": None,
    }

    try:
        pdb = app.PDBFile(_prepare_pdb(pdb_path))
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        # gbn2.xml is already loaded in ForceField; do not pass implicitSolvent
        # here — OpenMM 8.x raises if you specify both the XML and the flag.
        # HBonds constraints remove high-frequency H-X oscillations that blow
        # up the integrator when reconstructed sidechains have clashes.
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            hydrogenMass=1.5 * unit.amu,
        )
        integrator = omm.LangevinMiddleIntegrator(
            temp_K * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        omm.LocalEnergyMinimizer.minimize(
            simulation.context, maxIterations=n_steps_min
        )

        # A: PE gate — reject structures that are still severely clashed after
        # minimisation. Per-atom PE > 0 kJ/mol means repulsive clash energy
        # dominates; production MD at 310 K will blow up within picoseconds.
        state_min = simulation.context.getState(getEnergy=True)
        pe_min    = state_min.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        n_atoms   = system.getNumParticles()
        if pe_min / n_atoms > 0.0:
            raise ValueError(
                f"post-minimisation PE/atom = {pe_min/n_atoms:.1f} kJ/mol "
                f"(total {pe_min:.0f} kJ/mol) — structure still clashed, skipping"
            )

        # Save minimised structure as topology reference for later mdtraj loading
        with open(top_path, "w") as fh:
            app.PDBFile.writeFile(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                fh,
            )

        # B: Warm-up — 100 ps at 50 K before production to gently relax any
        # residual clash geometry the minimiser couldn't fully resolve.
        warmup_steps = int(0.1e6 / 2)   # 100 ps at 2 fs/step
        simulation.context.setVelocitiesToTemperature(50.0 * unit.kelvin)
        simulation.step(warmup_steps)
        integrator.setTemperature(temp_K * unit.kelvin)

        # Reporters: adaptive interval to ensure at least 100 frames
        n_steps = int(md_ns * 1e6 / 2)
        report_interval = max(1, n_steps // 100)
        simulation.reporters.append(app.DCDReporter(traj_path, report_interval))

        # Production MD
        simulation.step(n_steps)

        # Final potential energy
        state_f = simulation.context.getState(getPositions=True, getEnergy=True)
        pe = state_f.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Per-frame RMSD via mdtraj (already installed)
        import mdtraj as md
        traj = md.load(traj_path, top=top_path)
        ca_idx = traj.topology.select("name CA")
        ca_traj = traj.atom_slice(ca_idx)
        rmsd_nm = md.rmsd(ca_traj, ca_traj, frame=0)   # nm, relative to frame 0
        rmsd_A  = rmsd_nm * 10.0                        # → Å

        result.update({
            "final_pe_kJ":   round(float(pe), 2),
            "rmsd_initial_A": 0.0,
            "rmsd_final_A":  round(float(rmsd_A[-1]), 4),
            "rmsd_mean_A":   round(float(rmsd_A.mean()), 4),
            "rmsd_std_A":    round(float(rmsd_A.std()), 4),
            "stable": float(rmsd_A.std()) < 3.0 and float(rmsd_A[-1]) < 8.0,
            "error": None,
        })

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    with open(metrics_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result
