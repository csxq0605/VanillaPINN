"""Where does the field error come from? band_err.py <run_dir> [ckpt]
For each case: relL2 of rho/p overall, outside a band |d|<w around the exact shock, and the
'smearing floor' = relL2 of the exact solution smeared by a tanh profile with the PINN's own thickness
(i.e. what a perfectly placed shock of that thickness would score)."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN, evaluate_case, eval_cases, predict, oblique_shock, u_inf

run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else "model_final.pt"
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg)
m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
cpu = torch.device("cpu")
rows = []
for ma, td in eval_cases(cfg):
    th = math.radians(td); beta, post = oblique_shock(ma, th)
    qinf = torch.tensor([u_inf(ma), 0.0, 1.0, 1.0]); post = torch.tensor(post)
    xg, yg = torch.meshgrid(torch.linspace(cfg.x0, cfg.x1, 240), torch.linspace(0, cfg.y1, 180), indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1); ok = y >= x.clamp(min=0) * math.tan(th); x, y = x[ok], y[ok]
    q = predict(m, x, y, ma, th, cpu)
    inside = (x > 0) & (y < x * math.tan(beta))
    qe = torch.where(inside[:, None], post[None], qinf[None])
    # signed normal distance to the exact shock line (x>0); upstream of apex -> distance to apex
    d = torch.where(x > 0, (y * math.cos(beta) - x * math.sin(beta)), torch.sqrt(x * x + y * y))
    ev = evaluate_case(m, cfg, ma, td, cpu, nx=60, ny=45)
    T = ev["thickness"]; delta = T / (2 * math.atanh(0.8))
    s = 0.5 * (1 - torch.tanh(d / max(delta, 1e-6)))           # 1 behind the shock, 0 ahead
    s = torch.where(x > 0, s, torch.zeros_like(s))
    qs = qinf[None] + s[:, None] * (post - qinf)[None]
    r = lambda a, b, msk: ((a[msk] - b[msk]).norm(dim=0) / b[msk].norm(dim=0))
    allm = torch.ones_like(x, dtype=torch.bool)
    out = {w: d.abs() > w for w in (0.03, 0.06)}
    rows.append((ma, td, ev["beta"] - ev["beta_exact"], T, r(q, qe, allm), r(q, qe, out[0.03]), r(q, qe, out[0.06]), r(qs, qe, allm)))
    e = rows[-1]
    print(f"Ma {ma:.0f} th {td:4.1f} | dbeta {e[2]:+5.2f} T {T:.3f} | p: all {100*e[4][3]:5.1f}%  out.03 {100*e[5][3]:4.1f}%  out.06 {100*e[6][3]:4.1f}%  "
          f"floor {100*e[7][3]:4.1f}% | rho: all {100*e[4][2]:5.1f}%  out.06 {100*e[6][2]:4.1f}%  floor {100*e[7][2]:4.1f}%", flush=True)
mean = lambda k, i: 100 * sum(float(rw[k][i]) for rw in rows) / len(rows)
print(f"MEAN p: all {mean(4,3):.1f}%  outside 0.03 {mean(5,3):.1f}%  outside 0.06 {mean(6,3):.1f}%  smearing floor {mean(7,3):.1f}% | "
      f"rho: all {mean(4,2):.1f}%  outside 0.06 {mean(6,2):.1f}%  floor {mean(7,2):.1f}%")
