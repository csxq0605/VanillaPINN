"""Entropy p/rho^g relative to the exact post-shock value: just behind the shock and next to the wall, vs x;
plus the weighted strong residual vs distance from the apex. entropy_map.py <run_dir> [ckpt] [cases...]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import param_pinn as P
from physics import GAMMA as g, oblique_shock

run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
cfg = P.PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = P.ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
dev = torch.device("cpu")
cases = [(6.0, 15.0), (5.0, 15.0), (4.0, 10.0), (3.0, 20.0)]
xs_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
for ma, thd in cases:
    th = math.radians(thd); be, post = oblique_shock(ma, th)
    s_ex = post[3] / post[2] ** g
    behind, wall, ang = [], [], []
    for xs in xs_list:
        yy = torch.linspace(xs * math.tan(th), cfg.y1, 2000)
        q = P.predict(m, torch.full_like(yy, xs), yy, ma, th, dev)
        p, r = q[:, 3], q[:, 2]
        pw = p[:10].mean().item()
        i50 = int((p > 1 + 0.5 * (pw - 1)).nonzero().max())
        i90 = int((p > 1 + 0.9 * (pw - 1)).nonzero().max())
        # just behind: 1.5 x (50%-90% distance) below the 90% point, but not below the wall
        j = max(0, i90 - int(1.5 * max(1, i50 - i90)))
        s = p / r ** g / s_ex
        behind.append(s[j].item()); wall.append(s[1].item())
        ang.append(math.degrees(math.atan2(yy[i50].item(), xs)))
    print(f"Ma {ma:.0f} th {thd:4.1f} (beta_ex {math.degrees(be):.2f})")
    print("   x            " + " ".join(f"{x:6.2f}" for x in xs_list))
    print("   chord beta   " + " ".join(f"{a:6.2f}" for a in ang))
    print("   s behind     " + " ".join(f"{v:6.3f}" for v in behind))
    print("   s at wall    " + " ".join(f"{v:6.3f}" for v in wall))
# strong residual (scaled as in training) vs r from the apex, all parameters
d = json.load(open(os.path.join(run, "config.json")))
d.update(init_from="", rad_pool=1000, out_root=os.path.join(__import__("tempfile").gettempdir(), "param_diag_runs"))
T = P.ParamTrainer(P.PConfig(**d)); T.model.load_state_dict(m.state_dict())
X = T.interior(200000)
res, _ = T.residual(X)
rr = X[:, :2].norm(dim=1); r2 = (res.detach() ** 2).sum(1); w = T.tip_weight(X)
print("r band        | mean res^2  | weighted (tip)  | share of weighted loss")
tot = (r2 * w).sum()
for a, b in [(0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 2)]:
    k = (rr >= a) & (rr < b)
    print(f"[{a:.2f},{b:.2f})   | {r2[k].mean():.3e}  | {(r2[k] * w[k]).mean():.3e}       | {((r2[k] * w[k]).sum() / tot):.3f}")
