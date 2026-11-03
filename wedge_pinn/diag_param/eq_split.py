"""Per-equation share of the PDE loss: eq_split.py <run_dir> [ckpt]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import param_pinn as P
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
d = json.load(open(os.path.join(run, "config.json")))
d.update(init_from="", out_root=os.path.join(__import__("tempfile").gettempdir(), "param_diag_runs"), rad_pool=40000)
cfg = P.PConfig(**d)
T = P.ParamTrainer(cfg)
T.model.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True))
torch.manual_seed(1)
T.resample(with_rad=True)
res, _ = T.residual(T.X)
w = T.w_tip[:, None]
ms = ((res ** 2) * w).mean(0)
print("mean sq residual per eq (current scaling):", [f"{v:.3e}" for v in ms.tolist()], "sum", f"{ms.sum():.3e}")
us = math.sqrt(P.GAMMA) * T.X[:, 2]
ab = torch.stack([us, us * us + 1, us * us + 1, P.G1 * us + 0.5 * us ** 3], 1)   # absolute free-stream fluxes
rel = res * T.flux_scale(T.X) / ab
mr = ((rel ** 2) * w).mean(0)
print("relative to own free-stream flux:          ", [f"{v:.3e}" for v in mr.tolist()])
