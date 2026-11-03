"""Conservation check across the captured shock: rh_check.py <run_dir> [ckpt]
At x stations, find the local shock line (50% p crossing on verticals), take its normal n, and compare the
normal fluxes (mass, normal momentum, tangential velocity, total enthalpy) just behind the shock with the
free stream; also give the downstream state that RH would give for the captured local angle."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN, predict, u_inf
from physics import GAMMA, oblique_shock
g = GAMMA
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
dev = torch.device("cpu")

def shock_y(ma, th, xs):
    yy = torch.linspace(xs * math.tan(th), cfg.y1, 1500)
    p = predict(m, torch.full_like(yy, xs), yy, ma, th, dev)[:, 3]
    pw = p[: 20].mean().item()
    return yy[p > 1 + 0.5 * (pw - 1)].max().item()

def fluxes(q, n, t):
    u, v, r, p = q.unbind(-1)
    un = u * n[0] + v * n[1]; ut = u * t[0] + v * t[1]
    H = g / (g - 1) * p / r + 0.5 * (u * u + v * v)
    return torch.stack([r * un, p + r * un * un, ut, H], -1), torch.stack([un, ut, r, p], -1)

for ma in (3.0, 4.0, 5.0, 6.0):
    for thd in (5.0, 10.0, 15.0, 20.0):
        th = math.radians(thd); be, post = oblique_shock(ma, th)
        U = u_inf(ma); out = []
        for xs in (0.6, 0.85):
            y0, y1 = shock_y(ma, th, xs - 0.05), shock_y(ma, th, xs + 0.05)
            bl = math.atan2(y1 - y0, 0.1); ys = 0.5 * (y0 + y1)
            n = (math.sin(bl), -math.cos(bl)); t = (math.cos(bl), math.sin(bl))
            # distance from shock point to wall along n
            dwall = (ys - xs * math.tan(th)) * math.cos(th) / math.cos(bl - th)
            d = min(0.06, 0.6 * dwall)
            pts = torch.tensor([[xs + d * n[0], ys + d * n[1]], [xs - d * n[0], ys - d * n[1]]])
            q = predict(m, pts[:, 0], pts[:, 1], ma, th, dev)
            F, S = fluxes(q, n, t)
            Finf, _ = fluxes(torch.tensor([U, 0.0, 1.0, 1.0]), n, t)
            rel = (F[0] - Finf) / Finf.abs()
            # RH downstream for the captured local angle (normal shock with M1n = Ma sin bl)
            M1n = ma * math.sin(bl)
            r2 = (g + 1) * M1n ** 2 / ((g - 1) * M1n ** 2 + 2); p2 = 1 + 2 * g / (g + 1) * (M1n ** 2 - 1)
            out.append(f"x{xs:.2f} bl {math.degrees(bl):5.2f} d {d:.3f} | dF mass {100*rel[0]:+5.1f} mom {100*rel[1]:+5.1f} ut {100*rel[2]:+5.1f} H {100*rel[3]:+5.1f}"
                       f" | rho {S[0,2]:.3f}/RH {r2:.3f}/ex {post[2]:.3f} p {S[0,3]:.3f}/RH {p2:.3f}/ex {post[3]:.3f}")
        print(f"Ma {ma:.0f} th {thd:4.1f} beta_ex {math.degrees(be):5.2f} || " + " || ".join(out))
