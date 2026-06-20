# Boltzmann-Correct Loss and Evaluation Redesign

**Date:** 2026-06-20
**Status:** Approved
**Supersedes:** sections 7 and 10 of `2026-06-20-long-stride-md-residue-frame-demo-design.md`

## 1. Problem statement

The original demo used **conditional flow matching (CFM)** with MSE velocity regression. CFM with a single target per source frame pushes the network to predict the *conditional mean* `E[x_{t+τ} | x_t]`, not the conditional distribution `p(x_{t+τ} | x_t)`. When the true distribution is multimodal (different conformational states reachable from the same starting structure), CFM averages across them and produces a low-energy chimera that belongs to none. This violates the Boltzmann-distribution requirement.

The evaluation — geometry validity + mean pairwise CA-RMSD diversity — did not measure distributional correctness at all.

Additionally, the 1 µs MD trajectory does not fully cover the Boltzmann ensemble: dominant conformational basins are over-represented, rare-but-reachable states appear in < 1% of frames, and energy barriers > ~5 kT prevent crossing within the simulation timescale. A model trained naively on this trajectory inherits these biases.

This spec describes the full redesign addressing all three gaps.

## 2. Scope

**Changed:**
- Loss function and sampler (`model.py`): replace CFM with DDPM score matching
- Training loop (`demo.py`): add inverse-density reweighting and target augmentation
- Preprocessing (`data.py`): add frame-density weight computation
- Evaluation (`validation.py`): replace ensemble_overlap with distributional metrics

**Unchanged:**
- `NoiseSchedule` is a new `nn.Module` only — `FlowNet` architecture is identical; only what the network predicts changes
- `data.py` loading, frame construction, multi-lag pair generation
- `decoder.py` (build_structure, idealize, write_pdb)
- `featurize.py`, `geometry.py`
- CLI interface (new flags added, existing flags preserved with same defaults)

## 3. Loss redesign — DDPM score matching

### 3.1 `NoiseSchedule` (new `nn.Module` in `model.py`)

Cosine schedule over T steps. All arrays are registered as buffers so they move to device with `.to(device)`.

**Precomputed arrays (shape `[T]` each):**
- `alphas_bar`: `cos²(π/2 · (t/T + s₀) / (1 + s₀))`, `s₀ = 0.008` (cosine offset prevents singularity at t=0)
- `sqrt_alphas_bar`: `√ᾱ_t`
- `sqrt_one_minus_alphas_bar`: `√(1 − ᾱ_t)`
- `betas`: `1 − ᾱ_t / ᾱ_{t-1}`, clipped to `[0, 0.999]`
- `posterior_variance`: `β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)`, used in stochastic reverse

Constructor signature: `NoiseSchedule(T=200)`

### 3.2 `ddpm_loss(net, u_target, node_feats, edge_index, edge_feats, tau, schedule, pair_weights=None, sigma_aug=0.0)` → scalar

1. If `sigma_aug > 0`: `u_target = u_target + σ_aug · N(0, I)` (target augmentation, see §5.2)
2. Sample noise level per batch item: `t ~ Uniform{t_min..T-1}` where `t_min = max(1, T//20)` (= 10 for T=200) prevents training at near-zero noise where gradient variance is dominated by the augmentation term; with `t_min = 10` and cosine schedule, minimum training noise level is √(1−ᾱ₁₀) ≈ 0.09
3. Sample `ε ~ N(0, I)` same shape as `u_target`
4. Forward process: `noisy_u = sqrt_ab[t] · u_target + sqrt_1mab[t] · ε`
5. Network prediction: `pred_eps = net(noisy_u, t/T, node_feats, edge_index, edge_feats, tau)` — `s = t/T ∈ [0,1]`, same interface as CFM
6. Per-sample loss: `L_i = mean over (N,6) of (pred_eps_i − ε_i)²`
7. If `pair_weights` provided (shape `[B]`): `loss = mean(pair_weights · L_i)`, else `loss = mean(L_i)`

The `s = t/T` input to `FlowNet` now encodes noise level, not flow-time. The network learns the conditional score `∇ log p_t(u | x_t, τ)` at all noise levels simultaneously. No other change to `FlowNet` is needed.

### 3.3 `sample_ddpm(net, node_feats, edge_index, edge_feats, K, tau, schedule, steps=50, eta=1.0, sigma_init=1.0)` → `[K, N, 6]`

DDPM/DDIM unified reverse process over a strided subset of `steps` timesteps through T.

1. Start: `u_T ~ N(0, σ_init² · I)`, shape `[K, N, 6]`
2. For each step from T down to 0 (uniformly strided):
   - `eps_pred = net(u_t, t/T, ..., tau)` — one batched forward for all K
   - `u0_hat = (u_t − sqrt_1mab[t] · eps_pred) / sqrt_ab[t].clamp_min(1e-8)`
   - `dir_xt = sqrt(1 − ᾱ_{t-1} − η²σ_t²) · eps_pred`
   - `u_{t-1} = sqrt_ab[t-1] · u0_hat + dir_xt + η · σ_t · z`, `z ~ N(0, I)`
   - `σ_t = sqrt(posterior_variance[t])`
3. Return `u_0`

`eta=1.0` → full DDPM (maximum diversity, correct Boltzmann-stationary sampling). `eta=0.0` → DDIM (deterministic, faster exploration, less diverse). `sigma_init > 1.0` → broader prior for out-of-distribution exploration.

**Why the stochastic term matters for Boltzmann:** different noise draws `z` at each step route different samples toward different energy minima. Without `η · σ_t · z`, all K samples collapse deterministically to the same mode (same u0_hat from the same start). With it, the reverse process is a discrete Langevin integrator — the correct dynamical picture for thermal fluctuations.

### 3.4 Backward compatibility

`cfm_loss` and `sample` remain in `model.py` (unchanged) so existing tests continue to pass. New functions are additive.

## 4. Training loop changes (`demo.py`)

### 4.1 New parameters

| Parameter | Default | Purpose |
|---|---|---|
| `T_diff` | 200 | Number of DDPM noise levels |
| `diff_steps` | 50 | Reverse-process steps at inference |
| `eta` | 1.0 | DDPM stochasticity (1=full, 0=DDIM) |
| `sigma_init` | 1.0 | Prior scale for reverse process start |
| `sigma_aug` | 0.05 | Target augmentation noise (0 to disable) |
| `density_clip` | 10.0 | Maximum density weight relative to mean |

### 4.2 Modified `train()` signature

```python
def train(frames, taus, epochs, k, hidden, layers, lr,
          clip=1.0, batch_size=32, T_diff=200, sigma_aug=0.05,
          density_clip=10.0, device=None):
    ...
    schedule = m.NoiseSchedule(T=T_diff).to(device)
    frame_weights = data.compute_frame_weights(frames, density_clip=density_clip)
    pair_weights_all = frame_weights[train_pairs[:, 0]]  # [P]
    ...
    # In batch loop:
    pair_w = pair_weights_all[batch_indices].to(device)
    loss = m.ddpm_loss(net, u_batch, node_feats, edge_index, edge_feats,
                       tau_b, schedule, pair_weights=pair_w, sigma_aug=sigma_aug)
    ...
    return net, schedule, (node_feats, edge_index, edge_feats)
```

`sigma` parameter is removed from `train()` — it was CFM-specific. `NoiseSchedule` carries all noise level information.

### 4.3 Modified `run_demo()` inference

```python
u = m.sample_ddpm(net, node_feats, edge_index, edge_feats,
                  K=K, tau=infer_tau, schedule=schedule,
                  steps=diff_steps, eta=eta, sigma_init=sigma_init)
```

MD reference ensemble assembled from all val pairs matching `infer_tau` (up to `M=128`):
```python
matching_val = val_pairs[val_pairs[:, 2] == infer_tau]
ref_end_frames = matching_val[:, 1][:128]  # [M] end-frame indices
md_atoms = torch.stack([
    dec.build_structure(frames["R"][j].cpu(), frames["t"][j].cpu())
    for j in ref_end_frames
])  # [M, N, 4, 3]
```

This replaces the single `md_ca` comparison used previously.

## 5. Incomplete-sampling corrections

### 5.1 Inverse-density reweighting (`data.compute_frame_weights`)

```python
def compute_frame_weights(frames, n_pca=3, bins=30, density_clip=10.0):
    """Inverse-density weights for training pairs.

    Returns:
        weights [F]: float32 tensor, mean=1, max=density_clip
    """
```

Algorithm:
1. Extract CA coordinates: `ca = frames["t"]` → shape `[F, N, 3]`
2. Flatten and center: `ca_flat = ca.reshape(F, -1); ca_flat -= ca_flat.mean(0)`
3. SVD: `_, _, Vt = torch.linalg.svd(ca_flat, full_matrices=False)`; project to `[F, n_pca]`
4. 2D histogram in PC1–PC2 space with `bins×bins` cells
5. Map each frame to its bin count; weight = `1 / count`
6. Clip to `density_clip × mean_weight`; normalize to mean=1

Bin-count estimator is O(F) and avoids any pairwise computation. The `density_clip` cap prevents extreme weights for genuinely isolated frames from destabilizing training.

Effect: a frame in the dominant basin at 80% trajectory occupancy gets weight ~0.1; a frame in a sparse region visited 0.5% of the time gets weight ~2 (clipped). Over many epochs, the model sees all conformational regions roughly equally.

### 5.2 Target augmentation

Applied inside `ddpm_loss` before the forward diffusion process:

```python
if sigma_aug > 0:
    u_target = u_target + sigma_aug * torch.randn_like(u_target)
```

Default `sigma_aug = 0.05`. With typical `‖u_target‖ ≈ 0.3–1.0`, this is a ~5–15% perturbation. It smooths the discrete MD sample distribution into a continuous one, teaching the model that conformations *near* observed MD frames are also valid futures. Effectively sets a minimum noise floor at the lowest DDPM timesteps.

### 5.3 Inference temperature scaling

`sigma_init > 1.0` in `sample_ddpm` allows controlled exploration beyond the training distribution. With `sigma_init = 1.5`, the reverse process starts from a broader prior, traversing regions of conformation space the MD trajectory may not have visited. Useful for probing low-probability but physically accessible states.

This is an inference-time knob only — no retraining required.

## 6. Evaluation redesign (`validation.py`)

### 6.1 Functions removed

`ensemble_overlap` — replaced by more informative distributional metrics below.

### 6.2 `backbone_torsions(atoms [N, 4, 3])` → `(phi [N-2], psi [N-2])`

Computes backbone dihedral angles for interior residues using the four-atom cross-product formula. Atom ordering in `atoms`: `[N, CA, C, O]` at axis 1.

- `phi_i`: dihedral of `C(i-1)–N(i)–CA(i)–C(i)` (requires `atoms[i-1, 2]` = C of previous residue)
- `psi_i`: dihedral of `N(i)–CA(i)–C(i)–N(i+1)` (requires `atoms[i+1, 0]` = N of next residue)

Valid range: both angles `∈ (−π, π]`. Returns float tensors on the same device as `atoms`.

### 6.3 `ramachandran_js(atoms_model [K,N,4,3], atoms_md [M,N,4,3], bins=36)` → float ∈ [0, 1]

1. Compute `(φ_i, ψ_i)` for all K×(N−2) model residue-positions and all M×(N−2) MD residue-positions
2. Build 2D histogram on `[−π, π]² ` with `bins×bins` cells (10° resolution at bins=36) for each; add `ε = 1e-8` for numerical stability, then normalize to sum=1
3. Compute Jensen-Shannon divergence: `JS(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M)`, `M = ½(P+Q)`
4. Return `JS ∈ [0, 1]` (JS divergence is bounded by 1 in nats)

**Interpretation:** JS = 0 means identical Ramachandran distributions (perfect Boltzmann match); JS = 1 means completely disjoint (catastrophic failure). Values < 0.1 indicate good agreement for a sparse training set.

### 6.4 `pca_js(atoms_model [K,N,4,3], atoms_md [M,N,4,3], n_components=2, bins=20)` → dict

1. Fit PCA on MD CA coordinates (centers on mean MD structure)
2. Project both model and MD ensembles onto PC1–PC2
3. Build 2D density histograms over a shared axis range; compute JS divergence
4. Return `{"js": float, "var_explained": [float, float]}` where `var_explained` is fraction of MD variance on each PC

Captures whether the model explores the same large-scale conformational directions as MD (loop openings, helix tilts). Low `var_explained` values mean the model must compress information into fewer modes than MD.

### 6.5 `ensemble_recall(atoms_model [K,N,4,3], atoms_md [M,N,4,3], r_ang=2.0)` → float

Fraction of MD reference frames that have at least one model sample within `r_ang` Å CA-RMSD:

```python
covered = sum(
    1 for m in range(M)
    if min_k RMSD(ca_model[k], ca_md[m]) < r_ang
)
return covered / M
```

Measures: *does the model cover all conformational states the MD visits?*  
Low recall = mode collapse (model missing regions of conformation space).

### 6.6 `ensemble_novelty(atoms_model, atoms_md, r_ang=2.0)` → float

Fraction of model samples with no MD neighbor within `r_ang`:

```python
novel = sum(
    1 for k in range(K)
    if min_m RMSD(ca_model[k], ca_md[m]) >= r_ang
)
return novel / K
```

Measures: *does the model generalize beyond the training trajectory?*

### 6.7 Coverage interpretation table (embedded in docstring)

| recall | novelty | diagnosis |
|---|---|---|
| ≈ 1.0 | ≈ 0.0 | Faithful MD surrogate — correct coverage, no extrapolation |
| ≈ 1.0 | > 0.0 | Good generalization — covers MD + explores new states |
| < 0.8 | ≈ 0.0 | Mode collapse — missing conformational states |
| any | high + bad geometry | Hallucination — reduce `sigma_aug` or `sigma_init` |

### 6.8 Updated `run_demo` report keys

```python
report = {
    "model_geometry":     val.geometry_metrics(atoms_K[0]),
    "diversity_rmsd":     val.diversity(atoms_K),
    "ramachandran_js":    val.ramachandran_js(atoms_K, md_atoms),
    "pca_js":             pca_result["js"],          # pca_result = val.pca_js(...) computed once
    "pca_var_explained":  pca_result["var_explained"],
    "ensemble_recall":    val.ensemble_recall(atoms_K, md_atoms),
    "ensemble_novelty":   val.ensemble_novelty(atoms_K, md_atoms),
    "n_residues":         frames["R"].shape[1],
    "n_md_reference":     md_atoms.shape[0],
    "taus":               taus,
    "infer_tau":          infer_tau,
}
```

`ensemble_overlap_vs_true` is removed. `diversity` renamed to `diversity_rmsd` for clarity.

## 7. New CLI flags

```
--T_diff       int    200     DDPM noise levels
--diff_steps   int    50      Reverse-process steps at inference
--eta          float  1.0     DDPM stochasticity (1=full DDPM, 0=DDIM)
--sigma_init   float  1.0     Prior scale for reverse-process start
--sigma_aug    float  0.05    Target augmentation noise (0 to disable)
--density_clip float  10.0    Max density weight relative to mean
```

`--sigma` is removed (was CFM-specific). `--sigma_aug` replaces its role as the primary stochasticity-controlling hyperparameter at training time.

## 8. Test plan

### 8.1 `test_model.py` — new tests replacing CFM tests

| Test | What it checks |
|---|---|
| `test_noise_schedule_shape` | Buffer shapes `[T]`, values in [0,1] |
| `test_noise_schedule_monotone` | `sqrt_ab` decreasing, `sqrt_1mab` increasing |
| `test_ddpm_loss_unbatched` | Returns scalar > 0 with `[N, 6]` u_target |
| `test_ddpm_loss_batched` | Returns scalar > 0 with `[B, N, 6]` u_target |
| `test_ddpm_loss_weighted` | Weighted loss ≠ unweighted loss for unequal weights |
| `test_ddpm_can_overfit_constant_target` | 300 Adam steps, `mean‖samples − target‖ < 0.5` |
| `test_sample_ddpm_shape` | Returns `[K, N, 6]` |
| `test_sample_ddpm_diverse` | `samples.std(0).mean() > 0` |
| `test_sample_ddpm_eta0_deterministic` | Two calls with same seed + eta=0 give identical output |

### 8.2 `test_validation.py` — new tests

| Test | What it checks |
|---|---|
| `test_backbone_torsions_shape` | Returns `(N-2,)` for both phi and psi |
| `test_backbone_torsions_range` | All values ∈ `(−π, π]` |
| `test_ramachandran_js_identical` | JS = 0 when model and MD are the same ensemble |
| `test_ramachandran_js_disjoint` | JS ≈ 1 when distributions non-overlapping |
| `test_ensemble_recall_perfect` | recall = 1.0 when model exactly covers MD |
| `test_ensemble_recall_zero` | recall = 0 when model is far from all MD frames |
| `test_ensemble_novelty_zero` | novelty = 0 when model matches MD exactly |

### 8.3 `test_data.py` — new tests

| Test | What it checks |
|---|---|
| `test_compute_frame_weights_shape` | Returns `[F]` float tensor |
| `test_compute_frame_weights_mean` | Mean ≈ 1.0 |
| `test_compute_frame_weights_clip` | No weight exceeds `density_clip × mean` |
| `test_compute_frame_weights_uniform` | Identical frames → uniform weights |

### 8.4 `test_demo.py` — updated smoke test

`test_run_demo_smoke` updated:
- Assert `"ramachandran_js"` in report (not `"ensemble_overlap_vs_true"`)
- Assert `"ensemble_recall"` in report
- Assert `"ensemble_novelty"` in report
- Assert `"diversity_rmsd"` in report

## 9. Limitations and known gaps

**Incomplete sampling is attenuated, not eliminated.** Inverse-density reweighting corrects for over-represented basins within the sampled conformational space but cannot generate states the MD trajectory never visited. If a conformational state is completely absent from the 1 µs trajectory, the model will not learn it regardless of reweighting.

**Evaluation reference is the MD ensemble, not the true Boltzmann distribution.** `ramachandran_js = 0` means the model matches the MD distribution, which may itself differ from the true thermodynamic distribution due to force-field biases and finite simulation time. The metrics measure *MD-surrogate fidelity*, not absolute physical correctness.

**Single-chain trajectory.** The KRas-GDP system with 169 residues and 5001 frames at 200 ps/frame provides ~19,696 training pairs at τ ∈ {10, 25, 50, 100, 200}. Coverage of transitions > 40 ns (τ = 200 frames) will be sparse.

**Backbone only.** Side chains, which carry most of the binding-site chemical information, are not modeled. Ramachandran and PCA metrics are backbone-level only.
