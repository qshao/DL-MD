import math
import torch
import lsmd.cg_energy as cge


def reweight_boltzmann(traj: torch.Tensor,
                       res_type: torch.Tensor,
                       chain_id: torch.Tensor,
                       kT: float,
                       **energy_kwargs) -> dict:
    """Compute Boltzmann weights for a trajectory under the CG energy.

    Args:
        traj:      [F, N, 3] CA positions on CPU.
        res_type:  [N] long CANONICAL residue indices.
        chain_id:  [N] long chain assignment.
        kT:        thermal energy in kcal/mol (e.g. 0.593 at 300 K).
        **energy_kwargs: forwarded to total_cg_energy (w_wca, w_angle, w_mj, etc.)

    Returns:
        {"weights": Tensor[F] normalized, "n_eff": float, "degenerate": bool}
    """
    F = traj.shape[0]
    energies = torch.stack([
        cge.total_cg_energy(traj[i], res_type, chain_id, **energy_kwargs)
        for i in range(F)
    ])                                          # [F]
    log_w = -energies / kT
    log_w = log_w - log_w.max()                # numerical stability
    w = torch.exp(log_w)
    w = w / w.sum()
    n_eff = float(w.sum().pow(2) / w.pow(2).sum())
    return {"weights": w, "n_eff": n_eff, "degenerate": n_eff < 0.1 * F}


def resample_trajectory(traj: torch.Tensor,
                        weights: torch.Tensor,
                        n_samples: int = 500) -> torch.Tensor:
    """Resample trajectory frames by Boltzmann weights (with replacement).

    Args:
        traj:      [F, N, 3] CA positions.
        weights:   [F] non-negative weights (need not be normalized).
        n_samples: number of resampled frames.

    Returns:
        [n_samples, N, 3] resampled trajectory.
    """
    idx = torch.multinomial(weights, n_samples, replacement=True)
    return traj[idx]


def mh_rollout(net, sched, norm, R0, t0, res_type, chain_id, res_index, *,
               steps: int,
               tau_ps: float,
               k: int,
               diff_steps: int = 20,
               eta: float = 1.0,
               temp_K: float = 300.0,
               kT: float = 0.593,
               noether: bool = True,
               **energy_kwargs) -> torch.Tensor:
    """Metropolis-Hastings rollout for rigorous equilibrium sampling.

    Proposes each step via rollout(steps=1) and accepts/rejects via
    exp(-dU/kT). Rotation matrices R are approximated: R at each accepted
    step is derived from the previous rollout call (R does not accumulate
    across MH rejections -- this approximation affects proposal quality
    but not the acceptance criterion or the stationary distribution).

    Returns [steps+1, N, 3]. Library function only -- not CLI-exposed in Phase 2.
    """
    from lsmd import transfer_eval as te
    R = R0.clone()
    t = t0.clone()
    traj = [t.clone()]
    E_cur = cge.total_cg_energy(t, res_type, chain_id, **energy_kwargs)

    for _ in range(steps):
        prop = te.rollout(net, sched, norm, R, t, res_type, chain_id, res_index,
                          steps=1, tau_ps=tau_ps, k=k, diff_steps=diff_steps,
                          eta=eta, temp_K=temp_K, noether=noether)
        t_prop = prop[1]
        E_prop = cge.total_cg_energy(t_prop, res_type, chain_id, **energy_kwargs)
        dU = float(E_prop - E_cur)
        if dU <= 0 or torch.rand(1).item() < math.exp(-dU / kT):
            t = t_prop
            E_cur = E_prop
        traj.append(t.clone())

    return torch.stack(traj)
