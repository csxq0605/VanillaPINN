![banner](./assets/banner.png)

**English** | [中文](README_CN.md)

# VanillaPINN: Physics-Informed Neural Networks for Supersonic Wedge Flow with Oblique Shocks

## About

This repository collects the code and results of an **undergraduate research project at Peking University** on physics-informed neural networks (PINNs) for compressible flows with shocks.

- **Problem:** the attached oblique shock produced by supersonic flow over a wedge. The governing equations are the 2D steady inviscid compressible Euler equations (γ = 1.4).
- **Long-term target:** oblique detonation waves (ODW). The oblique shock is the lead structure of an ODW and a standard hard case for PINNs, because it is a discontinuity.
- **Core questions:**
  - **Accuracy.** A smooth network struggles to represent a discontinuity; early versions had shock-angle errors above 1°.
  - **Generalisation.** Can **one network** cover a whole range of Mach numbers and wedge angles instead of being trained per case?

The repository contains the solver (`wedge_pinn/`), the final model, the results and the documentation.

## Research Content

The project went through three stages:

| Stage | Content | Code |
|---|---|---|
| 1. Baseline | Fully connected PINN with pointwise learned artificial viscosity for Ma 5, 10° wedge, compared with OpenFOAM (early code, not included) | — |
| 2. Single-case high accuracy | Three steps: shock-capturing PINN with no shock prior → automatic shock detection from its solution → shock-fitting PINN refinement (explicit shock curve + RH condition) | `wedge_pinn/capture_lv.py`, `refine.py`, `shockfit.py` |
| 3. Parametric, equation-only | Inputs (x, y, Ma, θ); one model for Ma 3–6 × θ 5–20°. **No shock prior anywhere**: no RH condition, no shock position, no shock-aligned sampling — only the equations and how they are discretised | `wedge_pinn/param_pinn.py` |

Stage 3 is the main work. Each ingredient rests on a stated theoretical argument:
- **Vanishing-viscosity continuation:** a constant artificial viscosity is reduced step by step, so the RH conditions are satisfied automatically.
- **Parameter-modulated first layer:** neurons can rotate together with the shock surface.
- **Per-law flux scaling:** each conservation law is divided by its own free-stream flux, so mass, momentum and energy are weighted evenly.
- **Wedge-tip corner weighting:** handles the geometric singularity at the tip.
- **Augmented-Lagrangian wall condition:** removes the bias of the penalty method.
- **4D residual-adaptive sampling:** points concentrate on the shock surface by themselves.
- **Wider training box:** removes errors at the edges and corners of the parameter range.
- **Function-preserving widening:** adds network capacity starting from an existing solution.

See [Method](docs/method.md) (Chinese).

## Results

### 1. One network for the whole (Ma, θ) plane, equations only (`models/param_B1_w192`)

The model is compared with exact oblique-shock theory on 16 cases, Ma {3, 4, 5, 6} × θ {5°, 10°, 15°, 20°} ([per-case table](results/parametric/final_eval.md)):

| Metric | Mean | Max |
|---|---|---|
| Shock-angle error | **0.09°** | **0.17°** |
| ρ relative L2 | 3.0% | 5.2% |
| p relative L2 | 4.4% | 8.0% |

- The field error sits almost entirely in the thin shock band. Outside ±0.03 L of the shock it is **0.2%**.
- Of the 4.4% pressure error, 3.0 points come from the shock thickness itself: a perfectly placed shock of the same thickness would have that much error.
- The shock angle implied by the post-shock pressure is within 0.04° on average.
- The starting point was a plain MLP (shock angle 5.1° mean / 8.4° max, ρ 19%). The ablation figure below shows the path from there.

![error map](assets/error_map.png)

### 2. Single case: capture → detect → shock-fit (Ma 5, vs exact)

| Case | Shock angle (exact) | Perturbation L2 (u, v, ρ, p) |
|---|---|---|
| 10° wedge | 19.375–19.378° (19.376°) | 4.9e-4, 1.5e-4, 2.2e-4, 2.4e-4 |
| 15° wedge | 24.324–24.329° (24.322°) | 4.1e-4, 3.7e-5, 6.6e-4, 8.7e-5 |

- **Capture alone, no prior (10°):** relative L2 against OpenFOAM is ρ 1.0–1.3% and p 1.8–2.5% over 3 seeds. This is on par with the CFD's own error against the exact solution (ρ 1.5%, p 2.6%).
- **After refinement:** the remaining ~9% perturbation difference from the CFD equals the CFD's own numerical smearing of the shock.
- **Baseline:** the best early vanilla PINN reached 18.98° on the same case, and that model was selected by its CFD validation error.

### 3. Ablations

Each step on the main path changed one thing. All steps, and the attempts that did not help, are documented in [Ablations](docs/ablations.md) (Chinese). Attempts that did not help:
- learned viscosity
- polar coordinates
- mirror wall conditions
- snapshot averaging
- more collocation points
- lower viscosity

![ablation path](assets/ablation_path.png)

## Repository Contents

```
VanillaPINN/
├── wedge_pinn/                 # solver
│   ├── param_pinn.py           # stage 3: parametric equation-only PINN
│   ├── capture_lv.py           # stage 2: shock-capturing PINN
│   ├── refine.py, shockfit.py  # stage 2: automatic shock detection + shock-fitting refinement
│   ├── physics.py, model.py, data.py, evaluate.py   # exact theory, Euler residuals, network, sampling, metrics
│   ├── diag_param/             # diagnostics: conservation across the shock, error split, precise shock angle, corner scans
│   ├── scripts/train_param.sh  # full training chain of the final model
│   ├── make_figures.py         # figures and result table on this page
│   └── data/                   # cached OpenFOAM reference fields (10°, 15°)
├── models/param_B1_w192/       # final parametric model
├── results/                    # parametric 16-case results, single-case method comparison
└── docs/                       # method, ablations, literature survey
```

## Usage

Python 3.10 and PyTorch 2.5 (training on an RTX 4060 8 GB; evaluation also runs on CPU); see `requirements.txt`.

```bash
cd wedge_pinn
python diag_param/eval_param.py ../models/param_B1_w192   # evaluate the final model (CPU, ~5 min)
bash scripts/train_param.sh                                # train the parametric model from scratch (~12-14 h)
```

Single-case commands are in [Method](docs/method.md). Any config field can be overridden with `--field value`.

## Next Steps

- **Oblique detonation:** add a chemical-reaction source term. The lead shock is handled with this method.
- **Thinner shocks:** go below ν 7e-4, which needs more network capacity or resolution.
- **Parametric shock fitting:** condition the refinement networks and the shock curve on (Ma, θ).
