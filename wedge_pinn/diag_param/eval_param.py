"""Per-case table for a parametric run: eval_param.py <run_dir> [ckpt]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN, evaluate_grid, summary_line
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg)
m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True))
m.eval()
ev = evaluate_grid(m, cfg, torch.device("cpu"), nx=160, ny=120)
# mean nu along a line for each case (learned mode)
for e in ev:
    st = " ".join(f"{b:6.2f}" for b in e["beta_stations"])
    extra = ""
    if cfg.nu_mode == "learned":
        th = math.radians(e["theta"])
        xs = torch.linspace(-0.2, 1.0, 120); ys = torch.linspace(0, cfg.y1, 100)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        x, y = xg.reshape(-1), yg.reshape(-1); ok = y > x.clamp(min=0) * math.tan(th)
        X = torch.stack([x[ok], y[ok], torch.full_like(x[ok], e["ma"]), torch.full_like(x[ok], th)], 1)
        with torch.no_grad():
            m(X); s = m.extra[:, 0]; nu = cfg.nu + s * s
        inside = (X[:, 0] > 0) & (X[:, 1] < X[:, 0] * math.tan(math.radians(e["beta_exact"])) - 0.1)
        free = X[:, 1] > X[:, 0].clamp(min=0) * math.tan(math.radians(e["beta_exact"])) + 0.1
        extra = f" | nu max {nu.max():.3f} free {nu[free].mean():.4f} post {nu[inside].mean():.4f}"
    print(f"Ma {e['ma']:.0f} th {e['theta']:4.1f} | beta {e['beta']:6.2f} / {e['beta_exact']:6.2f} ({st}) | T {e['thickness']:.3f} | "
          f"rho {100*e['relL2'][2]:5.1f}% p {100*e['relL2'][3]:5.1f}% | post rho {100*e['post_err'][2]:+6.1f}% p {100*e['post_err'][3]:+6.1f}%{extra}")
print(summary_line(ev))
