"""Flow angle on the wall and just above it, vs the wedge angle: wall_angle.py <run> [ckpt]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN
from physics import oblique_shock
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
print("flow angle - theta (deg) at x = 0.2 / 0.5 / 0.8;  wall | 1/3 of the way to the exact shock | exact post p vs PINN p on wall")
for ma in (3.0, 5.0, 6.0):
    for thd in (5.0, 10.0, 15.0, 20.0):
        th = math.radians(thd); beta, post = oblique_shock(ma, th)
        xs = torch.tensor([0.2, 0.5, 0.8])
        out = []
        for frac in (0.0, 1 / 3):
            ys = xs * (math.tan(th) + frac * (math.tan(beta) - math.tan(th)))
            X = torch.stack([xs, ys, torch.full_like(xs, ma), torch.full_like(xs, th)], 1)
            with torch.no_grad():
                q = m(X)
            out.append(torch.rad2deg(torch.atan2(q[:, 1], q[:, 0])) - thd)
            if frac == 0.0:
                pw = q[:, 3]
        print(f"Ma {ma:.0f} th {thd:4.1f} | wall {' '.join(f'{a:+6.2f}' for a in out[0])} | 1/3 {' '.join(f'{a:+6.2f}' for a in out[1])}"
              f" | p wall {' '.join(f'{p:5.2f}' for p in pw)} exact {post[3]:.2f}")
