"""Entropy / viscosity along the wall and between wall and shock for a capture run (diagnostic)."""
import sys, os, json, math, torch
from physics import Freestream
from data import WedgeDomain
from model import WedgePINN

run = sys.argv[1]
cfg = json.load(open(os.path.join(run, "config.json")))
ls = cfg.get("length_scale", 1.0)
dom = WedgeDomain(cfg["x0"] / ls, cfg["x1"] / ls, cfg["y0"] / ls, cfg["y1"] / ls, cfg["w_start"] / ls, cfg["theta_deg"])
m = WedgePINN(dom.bounds, Freestream(cfg["mach"]), cfg["width"], cfg["depth"], cfg["adaptive"], param=cfg["param"],
              n_extra=1, normalize_inputs=cfg["normalize_inputs"]).cuda()
f = "model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt"
m.load_state_dict(torch.load(os.path.join(run, f), weights_only=True))
from physics import oblique_shock
beta, post = oblique_shock(cfg["mach"], dom.theta)
s_post = post[3] / post[2] ** 1.4
nw = cfg.get("nu_wall", 0)
print(f"{run} ({f});  columns: x | wall: s/s_post rho p | mid: s/s_post rho p nu | near shock (90%): s/s_post")
for fr in [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]:
    x = dom.w_start + fr * dom.L
    yw = dom.y0 + (x - dom.w_start) * math.tan(dom.theta)
    ys = dom.y0 + (x - dom.w_start) * math.tan(beta)
    X = torch.tensor([[x, yw + 1e-4], [x, 0.5 * (yw + ys)], [x, yw + 0.9 * (ys - yw)]], device="cuda")
    with torch.no_grad():
        q = m(X); nu = cfg.get("nu_min", 0.0) + m.extra[:, 0] ** 2
    if nw > 0:
        d = (X[:, 1] - yw) * math.cos(dom.theta)
        nu = nu * torch.tanh(d / nw) ** 2
    s = (q[:, 3] / q[:, 2] ** 1.4 / s_post).tolist()
    print(f" {x:6.3f} | {s[0]:.3f} {q[0,2]:.3f} {q[0,3]:.3f} | {s[1]:.3f} {q[1,2]:.3f} {q[1,3]:.3f} {nu[1]:.1e} | {s[2]:.3f}")
print(f" exact post-shock rho {post[2]:.3f} p {post[3]:.3f}")
