"""Per-equation residual (relative to each law's own free-stream flux, 1/L) along the shock normal.
foot_residual.py <run_dir> ma th_deg xs"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import param_pinn as P

run, ma, thd, xs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d = json.load(open(os.path.join(run, "config.json")))
d.update(init_from="", rad_pool=1000, eq_scale="own",
         out_root=os.path.join(__import__("tempfile").gettempdir(), "param_diag_runs"))
T = P.ParamTrainer(P.PConfig(**d))
T.model.load_state_dict(torch.load(os.path.join(run, "model_last.pt"), map_location="cpu", weights_only=True))
m, th, dev = T.model, math.radians(thd), torch.device("cpu")


def shock_y(x):
    yy = torch.linspace(x * math.tan(th), T.cfg.y1, 1500)
    p = P.predict(m, torch.full_like(yy, x), yy, ma, th, dev)[:, 3]
    return yy[p > 1 + 0.5 * (p[:20].mean().item() - 1)].max().item()


y0, y1 = shock_y(xs - 0.05), shock_y(xs + 0.05)
bl = math.atan2(y1 - y0, 0.1); ys = 0.5 * (y0 + y1)
n = (math.sin(bl), -math.cos(bl))
xi = torch.linspace(-0.3, 0.12, 22)
X = torch.stack([xs + xi * n[0], ys + xi * n[1], torch.full_like(xi, ma), torch.full_like(xi, th)], 1)
res, _ = T.residual(X)
ur = P.u_inf(T.cfg.ma_ref)
res = res.detach() / ur        # now relative to the own free-stream flux of each law (1/L)
print(f"Ma {ma} th {thd} x {xs}: residual / own free-stream flux (1/L): mass, x-mom, y-mom, energy")
for i in range(len(xi)):
    print(f"{xi[i]:+.3f}  " + "  ".join(f"{v:+.4f}" for v in res[i].tolist()))
