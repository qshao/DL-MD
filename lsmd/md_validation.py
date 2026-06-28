"""OpenMM implicit-solvent MD validation for all-atom protein structures."""
import json
import os

try:
    import openmm as omm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def run_md(pdb_path, out_dir, md_ns, temp_K=310.0, n_steps_min=1000):
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
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            implicitSolvent=app.GBn2,
            soluteDielectric=1.0,
            solventDielectric=78.5,
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

        # Save minimised structure as topology reference for later mdtraj loading
        with open(top_path, "w") as fh:
            app.PDBFile.writeFile(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                fh,
            )

        # Reporters: adaptive interval to ensure at least 100 frames
        n_steps = int(md_ns * 1e6 / 2)
        report_interval = max(1, n_steps // 100)
        simulation.reporters.append(app.DCDReporter(traj_path, report_interval))

        # Run MD: ns → steps at 2 fs/step
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
