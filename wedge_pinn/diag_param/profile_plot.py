"""Pressure along vertical lines, PINN vs exact, for a few cases: profile_plot.py <run_dir> [ckpt] [out.png]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from param_pinn import PConfig, ParamPINN, predict, oblique_shock

run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else "model_final.pt"
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, "profiles_p.png")
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
cases = [(3, 5), (3, 20), (4, 15), (5, 10), (6, 10), (6, 20)]
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, (ma, td) in zip(axs.flat, cases):
    th = math.radians(td); beta, post = oblique_shock(ma, th)
    for xs, c in ((0.25, "tab:blue"), (0.95, "tab:red")):
        yy = torch.linspace(xs * math.tan(th), cfg.y1, 600)
        p = predict(m, torch.full_like(yy, xs), yy, ma, th, torch.device("cpu"))[:, 3]
        pe = torch.where(yy < xs * math.tan(beta), torch.tensor(post[3]), torch.tensor(1.0))
        ax.plot(yy, p, c=c, lw=1.6, label=f"PINN x={xs}")
        ax.plot(yy, pe, c=c, lw=1, ls="--", label=f"精确 x={xs}")
    ax.set_title(f"Ma {ma}, θ {td}°（激波角精确 {math.degrees(beta):.2f}°）", fontsize=10)
    ax.set_xlabel("y"); ax.set_ylabel("p / p∞"); ax.grid(alpha=0.3)
axs.flat[0].legend(fontsize=8)
fig.suptitle("同一个网络、不同 (Ma, θ)：沿竖线的压力，PINN（实线）对精确解（虚线）", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=130); print(out)
