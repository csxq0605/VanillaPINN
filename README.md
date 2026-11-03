![VanillaPINN banner](./assets/banner.png)

[![English](https://img.shields.io/badge/English-README-blue?logo=github)](README.md)
[![中文](https://img.shields.io/badge/中文-README-green?logo=github)](README_CN.md)
[![Docs](https://img.shields.io/badge/Docs-Method%20%7C%20Ablations-red?logo=googledocs&logoColor=white)](docs/method.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white)](https://www.python.org)

VanillaPINN contains physics-informed neural networks for **supersonic wedge flow with oblique shocks**. It was built for an undergraduate research program at Peking University, and the long-term target is oblique detonation waves (ODW).

The main result is **one parametric network for the whole (Ma, θ) plane, trained from the equations alone.**
- The inputs are (x, y, Mach number, wedge angle).
- No Rankine–Hugoniot condition, shock position or shock-aligned sampling is used.
- Over Ma 3–6 × θ 5–20°, the shock-angle error is **0.09° mean and 0.17° max**.
- Away from the shock (outside ±0.03 L), the field error is **0.2%**.

> Shocks from the equations alone, for every Mach number and wedge angle at once.

# 🚀 Get Started

### Requirements

| Item | Used here |
|---|---|
| **Python** | 3.10 |
| **PyTorch** | 2.5.1 + CUDA 12 (CPU works for evaluation) |
| **GPU** | RTX 4060 8 GB. One training run at a time; a stage of 500 L-BFGS steps takes 35–70 min |
| **Other** | numpy, scipy, matplotlib (`pip install -r requirements.txt`) |

### Evaluate the released model (CPU, ~5 min)

```bash
git clone https://github.com/csxq0605/VanillaPINN.git
cd VanillaPINN/wedge_pinn
pip install -r ../requirements.txt
python diag_param/eval_param.py ../models/param_B1_w192        # 16-case table vs exact theory
python make_figures.py                                           # figures in assets/, table in results/parametric/
```

### Query any (Ma, θ)

```python
import json, math, torch
from param_pinn import PConfig, ParamPINN, predict
cfg = PConfig(**json.load(open("../models/param_B1_w192/config.json")))
net = ParamPINN(cfg); net.load_state_dict(torch.load("../models/param_B1_w192/model_final.pt", map_location="cpu")); net.eval()
x, y = torch.rand(1000), 0.5 * torch.rand(1000)
q = predict(net, x, y, 4.5, math.radians(12.5), torch.device("cpu"))   # columns u, v, rho, p (rho_inf = p_inf = 1)
```

### Train

```bash
cd wedge_pinn
bash scripts/train_param.sh            # parametric model, full warm-start chain (about 12-14 h on one 8 GB GPU)
python capture_lv.py --width 64 --depth 6 --normalize_inputs true --nu_wall 0.05 --n_rad 10000 \
       --rad_every 200 --rad_until 400 --w_weak 10 --adam_iters 500 --nu_min 2e-3 --tag lvf   # single case: capture
python refine.py --capture runs/<capture run> --tag refine                                     # detect + shock-fit
```

Every config field can be overridden with `--field value`; see the dataclasses at the top of `param_pinn.py`, `capture_lv.py` and `shockfit.py`.

# ✨ Core Features

| Feature | Description |
|---|---|
| **🧮 Equation-only shock capturing** | Viscous Euler equations with a vanishing constant artificial viscosity (4e-3 → 7e-4). The RH jump conditions come out of the vanishing-viscosity limit; they are never imposed. |
| **🎛️ Parameter-modulated first layer** | A small network of (Ma, θ) generates the first-layer weights. Neurons can then rotate with the shock surface y = x·tanβ(Ma, θ), which is not a plane in input space (shift-DeepONet / HyperPINN idea). |
| **⚖️ Per-law flux scaling** | Each conservation law is divided by its own free-stream flux. Unscaled, the energy equation outweighs the mass equation by about 440×. This single change cut the mean shock-angle error from 0.76° to 0.48°. |
| **📐 Wedge-tip corner weighting** | Residuals are multiplied by r²/(r²+δ²), a Kondratiev-type weighted norm for the geometric singularity at the tip. It uses geometry only. |
| **🔗 Augmented-Lagrangian wall condition** | Removes the penalty bias that traded wall accuracy for a weaker shock. Wall under-turning dropped from 0.6–2.6° to 0.1–0.7°. |
| **🎯 4D residual-adaptive sampling** | Collocation points are resampled by residual in (x, y, Ma, θ), so they concentrate on the shock surface with no shock information. |
| **🧱 Wider training box + function-preserving widening** | Training on a parameter box wider than the evaluated range removes edge and corner errors. `widen_model.py` grows the network without changing its output (< 4e-7). |
| **🔬 Capture → detect → shock-fit pipeline** | For a single case: capture the shock, detect it automatically, then refine with an explicit shock curve. The shock-angle error is ≤ 0.007° and the perturbation L2 vs exact is ≤ 7e-4. |
| **🩺 Diagnostics** | `wedge_pinn/diag_param/` has 16 scripts: RH check across the shock, error inside/outside the shock band and thickness floor, precise shock angle, corner scans, 2D error/entropy/residual maps, and more. |

# 📊 Results

**Parametric model** (`models/param_B1_w192`, 16 cases vs exact theory, window y ≤ 0.9; [full table](results/parametric/final_eval.md)):

| | Mean | Max |
|---|---|---|
| Shock-angle error | 0.09° | 0.17° |
| ρ relative L2 | 3.0% | 5.2% |
| p relative L2 | 4.4% | 8.0% |

Of the 4.4% pressure L2, **3.0 points is the thickness floor**: the error a perfectly placed shock of the same thickness would have. Outside ±0.03 L of the shock, the error is 0.2%. The shock angle implied by the post-shock pressure is within 0.04° on average.

![error map](assets/error_map.png)

**Single case, Ma 5** (capture → detect → shock-fit):

| Case | Shock angle (exact) | Perturbation L2 vs exact (u, v, ρ, p) |
|---|---|---|
| 10° wedge | 19.375–19.378° (19.376°) | 4.9e-4, 1.5e-4, 2.2e-4, 2.4e-4 |
| 15° wedge | 24.324–24.329° (24.322°) | 4.1e-4, 3.7e-5, 6.6e-4, 8.7e-5 |

Capture alone (10°, no prior) gives ρ 1.0–1.3% and p 1.8–2.5% relative L2 against OpenFOAM over 3 seeds.

# 🧪 Ablations

The parametric model was built as a chain of warm-started stages, each changing one thing. The figure tracks mean |β error| and mean ρ L2 along the main path. The full tables, including the experiments that did not help, are in **[docs/ablations.md](docs/ablations.md)**. Experiments that did not help:
- learned viscosity
- polar coordinates
- mirror wall conditions
- snapshot averaging
- more RAD points
- ν 5e-4 at both network widths

![ablation path](assets/ablation_path.png)

# 📁 Repository Layout

```
VanillaPINN/
├── wedge_pinn/                 # main code
│   ├── param_pinn.py           # parametric equation-only PINN (x, y, Ma, θ)
│   ├── capture_lv.py           # single-case shock-capturing PINN
│   ├── refine.py, shockfit.py  # automatic shock detection + shock-fitting refinement
│   ├── physics.py, model.py, data.py, evaluate.py
│   ├── widen_model.py          # function-preserving widening
│   ├── make_figures.py         # figures / tables for the released model
│   ├── diag_param/             # diagnostics for parametric runs
│   ├── scripts/train_param.sh  # full training chain of the released model
│   ├── data/                   # cached OpenFOAM reference fields (10°, 15°) and the Fluent point set
│   └── capture.py, train.py, main.py, diag_*.py, compare.py, summarize.py   # baselines and single-case tools
├── models/param_B1_w192/       # released checkpoint (config.json = evaluation window, train_config.json = training)
├── results/                    # parametric 16-case table, single-case comparison tables and figures
├── docs/                       # method, ablations, literature survey (Chinese)
├── assets/                     # README figures
└── legacy/vanilla_fnn/         # the original vanilla PINN code of this repository
```

# 📖 Documentation

- **[Method](docs/method.md)**: problem setup, the two pipelines, and the reasoning behind every ingredient (Chinese)
- **[Ablations](docs/ablations.md)**: all ablation tables and negative results (Chinese)
- **[Literature survey](docs/literature_survey_cn.md)**: PINN shock methods, split into equation-only and strong-prior families (Chinese)

# 🗺️ What's Next

- **Oblique detonation waves:** add species or reaction progress with a source term (one-step Arrhenius or two-step induction–heat release). The lead shock is the oblique shock solved here.
- **Thinner shocks:** go below ν 7e-4. This needs more capacity or resolution than tested so far.
- **Parametric shock fitting:** condition the refinement networks and the shock curve on (Ma, θ).
