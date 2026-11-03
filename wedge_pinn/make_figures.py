"""Figures and result table for the released parametric model.

    python make_figures.py [model_dir]        (default ../models/param_B1_w192, runs on CPU in ~5 min)

Writes ../assets/banner.png, ../assets/fields_grid.png, ../assets/error_map.png, ../assets/ablation_path.png
and ../results/parametric/final_eval.md (16 cases vs the exact oblique-shock solution, evaluation window y <= 0.9).
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from param_pinn import PConfig, ParamPINN, evaluate_case, oblique_shock, predict  # noqa: E402

ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
RES = os.path.join(ROOT, "results", "parametric")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

mdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "param_B1_w192")
cfg = PConfig(**json.load(open(os.path.join(mdir, "config.json"))))
model = ParamPINN(cfg)
model.load_state_dict(torch.load(os.path.join(mdir, "model_final.pt"), map_location="cpu", weights_only=True))
model.eval()
cpu = torch.device("cpu")


def pressure_field(ma, td, nx=420, ny=300):
    th = math.radians(td)
    xg, yg = torch.meshgrid(torch.linspace(-0.05, cfg.x1, nx), torch.linspace(0, 0.9, ny), indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    p = predict(model, x, y, ma, th, cpu)[:, 3]
    p[y < x.clamp(min=0) * math.tan(th)] = float("nan")
    return p.reshape(nx, ny).T.numpy()


# ---------------------------------------------------------------- banner: one network, several (Ma, theta)
cases = [(3.0, 20.0), (3.5, 12.5), (4.5, 17.0), (5.0, 10.0), (6.0, 15.0)]
fig, axs = plt.subplots(1, len(cases), figsize=(16, 3.6), facecolor="#0d1117")
for ax, (ma, td) in zip(axs, cases):
    p = pressure_field(ma, td)
    be, _ = oblique_shock(ma, math.radians(td))
    ax.imshow(p, origin="lower", extent=[-0.05, cfg.x1, 0, 0.9], cmap="magma", aspect="auto")
    ax.plot([0, cfg.x1], [0, cfg.x1 * math.tan(be)], color="#7ee0ff", lw=0.8, ls="--")
    ax.fill_between([0, cfg.x1], [0, 0], [0, cfg.x1 * math.tan(math.radians(td))], color="#30363d")
    ax.set_title(f"Ma {ma:g}, θ {td:g}°", color="#e6edf3", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#30363d")
fig.suptitle("One equation-only PINN for the whole (Ma, θ) plane — pressure, dashed = exact shock",
             color="#e6edf3", fontsize=14, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(ASSETS, "banner.png"), dpi=110, facecolor=fig.get_facecolor())
plt.close(fig)
print("banner done", flush=True)

# ---------------------------------------------------------------- 16-case evaluation
mas, ths = [3.0, 4.0, 5.0, 6.0], [5.0, 10.0, 15.0, 20.0]
ev = {(m, t): evaluate_case(model, cfg, m, t, cpu) for m in mas for t in ths}
B = np.array([[ev[m, t]["beta"] - ev[m, t]["beta_exact"] for t in ths] for m in mas])
R = np.array([[100 * ev[m, t]["relL2"][2] for t in ths] for m in mas])
P = np.array([[100 * ev[m, t]["relL2"][3] for t in ths] for m in mas])
lines = ["| Ma | θ | β PINN | β exact | error | ρ rel. L2 | p rel. L2 | shock thickness (L) |", "|---|---|---|---|---|---|---|---|"]
for m in mas:
    for t in ths:
        e = ev[m, t]
        lines.append(f"| {m:g} | {t:g}° | {e['beta']:.2f}° | {e['beta_exact']:.2f}° | {e['beta'] - e['beta_exact']:+.2f}° | "
                     f"{100 * e['relL2'][2]:.1f}% | {100 * e['relL2'][3]:.1f}% | {e['thickness']:.3f} |")
summ = (f"Mean |β error| {np.abs(B).mean():.2f}°, max {np.abs(B).max():.2f}°; "
        f"mean ρ rel. L2 {R.mean():.1f}%, mean p rel. L2 {P.mean():.1f}%.")
open(os.path.join(RES, "final_eval.md"), "w", encoding="utf-8").write(
    "# Released model `models/param_B1_w192` vs exact oblique-shock theory\n\n"
    "Evaluation window x ∈ [-0.25, 1], y ∈ [0, 0.9]; β = mean 50%-pressure chord angle at x = 0.5 / 0.75 / 0.95.\n\n"
    + "\n".join(lines) + "\n\n" + summ + "\n")
print(summ, flush=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, A, title, cmap, sym in [(axs[0], B, "shock-angle error (deg)", "RdBu_r", True),
                                (axs[1], R, "ρ relative L2 (%)", "viridis", False),
                                (axs[2], P, "p relative L2 (%)", "viridis", False)]:
    kw = dict(vmin=-np.abs(A).max(), vmax=np.abs(A).max()) if sym else {}
    im = ax.imshow(A, origin="lower", cmap=cmap, **kw)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{A[i, j]:+.2f}" if sym else f"{A[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    color="k" if sym or A[i, j] > A.mean() else "w")
    ax.set_xticks(range(4), [f"{t:g}°" for t in ths]); ax.set_yticks(range(4), [f"{m:g}" for m in mas])
    ax.set_xlabel("wedge angle θ"); ax.set_ylabel("Ma"); ax.set_title(title)
    fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "error_map.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(4, 4, figsize=(14, 11))
for i, m in enumerate(mas):
    for j, t in enumerate(ths):
        ax = axs[i, j]; e = ev[m, t]
        ax.imshow(pressure_field(m, t, 300, 220), origin="lower", extent=[-0.05, cfg.x1, 0, 0.9], cmap="viridis", aspect="auto")
        be = math.radians(e["beta_exact"])
        ax.plot([0, cfg.x1], [0, cfg.x1 * math.tan(be)], "w--", lw=0.6)
        ax.set_title(f"Ma {m:g}, θ {t:g}°: β {e['beta']:.2f} / {e['beta_exact']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Pressure from the single parametric network (white dashed: exact shock; β PINN / exact)")
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "fields_grid.png"), dpi=100); plt.close(fig)

# ---------------------------------------------------------------- main training path (README table)
path = [("p1 MLP, learned ν", 5.12, 19.4), ("const ν 1e-2", 2.15, 27.0), ("hyper first layer", 2.54, 25.4),
        ("strong-Wolfe L-BFGS", 1.64, 15.7), ("w_bc 1→10", 0.73, 9.5), ("wall ALM", 0.74, 8.2),
        ("eq_scale own", 0.48, 7.1), ("ν 3e-3, δ 0.01", 0.24, 5.3), ("wide param box", 0.24, 5.3),
        ("ν 2.5e-3", 0.21, 5.0), ("ν 2e-3", 0.17, 4.5), ("ν 1.5e-3", 0.12, 4.0), ("ν 1e-3", 0.12, 3.7),
        ("ν 7e-4", 0.12, 3.5), ("RAD 22k", 0.08, 2.9), ("width 192 (final)", 0.09, 3.0)]
fig, ax1 = plt.subplots(figsize=(12, 4.4))
xs = np.arange(len(path))
ax1.semilogy(xs, [p[1] for p in path], "o-", color="#d62728", label="mean |β error| (deg)")
ax2 = ax1.twinx()
ax2.plot(xs, [p[2] for p in path], "s-", color="#1f77b4", label="mean ρ rel. L2 (%)")
ax1.set_xticks(xs, [p[0] for p in path], rotation=40, ha="right", fontsize=9)
ax1.set_ylabel("mean |β error| (deg, log)", color="#d62728"); ax2.set_ylabel("mean ρ rel. L2 (%)", color="#1f77b4")
ax1.set_title("Cumulative ablation path of the parametric model (16 cases, Ma 3–6 × θ 5–20°)")
ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "ablation_path.png"), dpi=110); plt.close(fig)
print("figures written to", ASSETS)
