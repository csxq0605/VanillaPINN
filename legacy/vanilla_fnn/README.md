# Legacy: vanilla PINN (2025)

The original code of this repository: a plain fully connected PINN (`FNN.py`) with pointwise learned viscosity for
the Ma 5, 10° wedge, trained against OpenFOAM data (`dataloader.py`, `train.py`, `utils.py`, `main.py`).
It is kept for reference; `wedge_pinn/` is the rewritten version. On the same case the rewrite improves the shock
angle from 18.98° (best legacy model, selected by CFD validation error) to 19.40° by capture alone and 19.376° after
shock fitting (exact 19.376°), see `results/single_case/table.md`.
