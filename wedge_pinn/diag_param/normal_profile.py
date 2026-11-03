"""Profile along the local shock normal: normal_profile.py <run_dir> ma th_deg xs"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import param_pinn as P
from physics import GAMMA as g, oblique_shock

run, ma, thd, xs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
cfg = P.PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = P.ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, "model_last.pt"), map_location="cpu", weights_only=True)); m.eval()
th = math.radians(thd); dev = torch.device("cpu")
be, post = oblique_shock(ma, th)

def shock_y(x):
    yy = torch.linspace(x * math.tan(th), cfg.y1, 1500)
    p = P.predict(m, torch.full_like(yy, x), yy, ma, th, dev)[:, 3]
    pw = p[:20].mean().item()
    return yy[p > 1 + 0.5 * (pw - 1)].max().item()

y0, y1 = shock_y(xs - 0.05), shock_y(xs + 0.05)
bl = math.atan2(y1 - y0, 0.1); ys = 0.5 * (y0 + y1)
n = (math.sin(bl), -math.cos(bl)); t = (math.cos(bl), math.sin(bl))
dwall = (ys - xs * math.tan(th)) * math.cos(th) / math.cos(bl - th)
print(f"Ma {ma} th {thd} x {xs}: local shock angle {math.degrees(bl):.2f} (exact {math.degrees(be):.2f}), dist to wall {dwall:.3f}")
xi = torch.linspace(float(sys.argv[5]) if len(sys.argv) > 5 else -0.06, dwall, 60)
X = torch.stack([xs + xi * n[0], ys + xi * n[1], torch.full_like(xi, ma), torch.full_like(xi, th)], 1)
q, dq, _ = m.forward_derivs(X, second=False)
u, v, r, p = q.detach().unbind(1)
un = u * n[0] + v * n[1]; ut = u * t[0] + v * t[1]
rn = (dq[0, :, 2] * n[0] + dq[1, :, 2] * n[1]).detach()
nu = cfg.nu * math.sqrt(g) * ma / P.u_inf(cfg.ma_ref)
U = P.u_inf(ma); mn_inf = U * math.sin(bl)
s = p / r ** g
print("  xi     rho    p     ut/Uinf  mass flux/inf  visc part  entropy/post_exact")
for i in range(0, 60, 3):
    print(f"{xi[i]:+.3f} {r[i]:6.3f} {p[i]:6.3f} {ut[i] / U:7.4f}  {r[i] * un[i] / mn_inf:8.4f}  {-nu * rn[i] / mn_inf:+.4f}  "
          f"{s[i] / (post[3] / post[2] ** g):.4f}")
