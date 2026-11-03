"""Checks for the control-volume loss: test_weak.py <run_dir>"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import param_pinn as P

run = sys.argv[1]
d = json.load(open(os.path.join(run, "config.json")))
d.update(init_from="", rad_pool=40000, eq_scale="own", w_weak=1.0,
         out_root=os.path.join(__import__("tempfile").gettempdir(), "param_diag_runs"))
cfg = P.PConfig(**d)
T = P.ParamTrainer(cfg)
T.model.load_state_dict(torch.load(os.path.join(run, "model_last.pt"), map_location="cpu", weights_only=True))
torch.manual_seed(3)

# 1. first derivatives identical with and without the second-derivative sweep
X = T.interior(500)
q1, dq1, d2 = T.model.forward_derivs(X)
q2, dq2, none = T.model.forward_derivs(X, second=False)
print("1. max |dq diff|", (dq1 - dq2).abs().max().item(), "d2 None:", none is None)

# 2. divergence theorem: box balance = h * mean residual over the square (fine 2D Gauss)
T.resample(with_rad=True)
B = T.boxes[torch.randperm(T.boxes.shape[0])[:6]]
bal = T.box_balance(B)
gx, gw = np.polynomial.legendre.leggauss(40)
gx, gw = torch.tensor(gx / 2, dtype=torch.float32), torch.tensor(gw / 2, dtype=torch.float32)
for i in range(B.shape[0]):
    xc, yc, ma, th, h = B[i].tolist()
    xs, ys = torch.meshgrid(xc + h * gx, yc + h * gx, indexing="ij")
    W = (gw[:, None] * gw[None, :]).reshape(-1)
    Xb = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.full((1600,), ma), torch.full((1600,), th)], 1)
    res, _ = T.residual(Xb)
    area = h * (res * W[:, None]).sum(0)
    print(f"2. box h {h:.3f} at ({xc:.3f},{yc:.3f}) Ma {ma:.2f} th {math.degrees(th):.1f}: "
          f"boundary {np.array2string(bal[i].detach().numpy(), precision=4)}  area {np.array2string(area.detach().numpy(), precision=4)}")

# 3. loss sizes with the current model
total, parts = T.loss()
print("3. parts", {k: f"{v:.3e}" for k, v in parts.items()})
bb = (T.box_balance(T.boxes) ** 2).sum(1)
na = int(cfg.box_adapt * cfg.n_box)
print(f"   box loss adaptive {bb[:na].mean():.3e} uniform {bb[na:].mean():.3e}; per-eq rms (adaptive)",
      (T.box_balance(T.boxes[:na]) ** 2).mean(0).sqrt().tolist())
