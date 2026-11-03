"""Compare the old vanilla PINNs, the new shock-capturing PINN and the shock-fitting PINN
against the OpenFOAM solution (10 deg, Ma 5 case).

    python compare.py --capture runs/<capturing run> --refine runs/<refine run> --shockfit runs/<shock-fitting run>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import Freestream, oblique_shock, exact_solution
from model import WedgePINN
from data import WedgeDomain, load_reference
import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.environ.get("VANILLA_PINN_DIR", os.path.join(HERE, "..", "..", "vanilla_Pinn"))  # early vanilla PINN checkpoints, not in the repo
dev = "cuda" if torch.cuda.is_available() else "cpu"


class OldFNN(torch.nn.Module):
    """vanilla_Pinn/FNN.py: (x, y) -> (u, v, exp rho, exp p, nu)"""

    def __init__(self, sd):
        super().__init__()
        n = len([k for k in sd if k.endswith(".weight")])
        self.layers = torch.nn.ModuleList(torch.nn.Linear(*sd[f"layers.{i}.weight"].shape[::-1]) for i in range(n))
        self.load_state_dict(sd)

    def forward(self, X):
        h = X
        for l in self.layers[:-1]:
            h = torch.tanh(l(h))
        o = self.layers[-1](h)
        return torch.stack([o[:, 0], o[:, 1], torch.exp(o[:, 2]), torch.exp(o[:, 3])], 1)


class OldGenerator(torch.nn.Module):
    """VanillaPinn.ipynb Generator: (x, y) -> (exp rho, exp p, u, v, nu); reordered to (u, v, rho, p)"""

    def __init__(self, sd):
        super().__init__()
        w = sd["map.0.weight"].shape[0]
        self.map = torch.nn.Sequential(torch.nn.Linear(2, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(),
                                       torch.nn.Linear(w, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(),
                                       torch.nn.Linear(w, 5))
        self.load_state_dict(sd)

    def forward(self, X):
        o = self.map(X)
        return torch.stack([o[:, 2], o[:, 3], torch.exp(o[:, 0]), torch.exp(o[:, 1])], 1)


class ShockFitModel:
    def __init__(self, run):
        from shockfit import ShockCurve
        cfg = json.load(open(os.path.join(run, "config.json")))
        fs = Freestream(cfg["mach"])
        dom = WedgeDomain(cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"], cfg["w_start"], cfg["theta_deg"])
        sd = torch.load(os.path.join(run, "model_final.pt"), weights_only=True, map_location=dev)
        self.up = WedgePINN(dom.bounds, fs, cfg["width"], cfg["depth"]).to(dev); self.up.load_state_dict(sd["up"])
        self.down = WedgePINN(dom.bounds, fs, cfg["width"], cfg["depth"]).to(dev); self.down.load_state_dict(sd["down"])
        self.shock = ShockCurve(cfg["w_start"], cfg["y0"], cfg["x1"], 0.3, cfg["curved"]).to(dev)
        self.shock.load_state_dict(sd["shock"])
        self.w_start = cfg["w_start"]

    def __call__(self, X):
        down = (X[:, 0] > self.w_start) & (X[:, 1] < self.shock(X[:, 0]))
        q = self.up(X)
        q[down] = self.down(X[down])
        return q


def load_capture(run):
    from refine import load_capture as lc
    return lc(run)[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="")
    ap.add_argument("--refine", default="")
    ap.add_argument("--shockfit", default="")
    ap.add_argument("--out", default="./comparison")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fs = Freestream(5.0)
    dom = WedgeDomain()
    beta, post = oblique_shock(5.0, dom.theta)
    q_inf = torch.tensor(fs.state(), device=dev)
    X, Q = load_reference(os.path.join(OLD, "reference", "program_25_11_16"), "0.005",
                          os.path.join(HERE, "data", "reference_cache.npz"), dev)
    Qex = exact_solution(X[:, 0], X[:, 1], fs, dom.w_start, dom.theta)

    models = {}
    p1 = os.path.join(OLD, "logs", "2025-11-18_17-32-39", "best_model.pth")
    if os.path.exists(p1):
        models["old FNN (2025-11-18)"] = OldFNN(torch.load(p1, weights_only=True, map_location=dev)["model_state_dict"]).to(dev)
    p2 = os.path.join(OLD, "logs", "2025-12-02_20-44-01", "generator_best.pth")
    if os.path.exists(p2):
        models["old notebook (2025-12-02)"] = OldGenerator(torch.load(p2, weights_only=True, map_location=dev)).to(dev)
    if args.capture:
        models["new capturing (no prior)"] = load_capture(args.capture)
    if args.refine:
        models["capture -> detect -> fit"] = ShockFitModel(args.refine)
    if args.shockfit:
        models["shock-fitting (hand-set topology)"] = ShockFitModel(args.shockfit)

    roi = dom.roi_mask(X[:, 0], X[:, 1])
    rows = []
    preds = {}
    for name, m in models.items():
        with torch.no_grad():
            P = torch.cat([m(X[i:i + 65536]) for i in range(0, X.shape[0], 65536)])
        preds[name] = P
        prof = ev.shock_profile(m, dom, post[3], device=dev)
        ang = [a for (_, _, a, _) in prof if np.isfinite(a)]
        rows.append(dict(method=name,
                         pertL2_cfd=ev.perturbation_l2(P, Q, q_inf, roi).tolist(),
                         pertL2_exact=ev.perturbation_l2(P, Qex, q_inf, roi).tolist(),
                         relL2_full=ev.rel_l2(P, Q).tolist(),
                         beta=float(np.mean(ang)) if ang else float("nan")))
    rows.append(dict(method="CFD (OpenFOAM, 400x400)",
                     pertL2_cfd=[0, 0, 0, 0],
                     pertL2_exact=ev.perturbation_l2(Q, Qex, q_inf, roi).tolist(),
                     relL2_full=[0, 0, 0, 0], beta=float("nan")))

    lines = ["| method | disturbance L2 vs CFD (u, v, rho, p) | disturbance L2 vs exact (u, v, rho, p) | shock angle [deg] |",
             "|---|---|---|---|"]
    for r in rows:
        f = lambda a: ", ".join(f"{v:.3f}" if abs(v) >= 1e-2 or v == 0 else f"{v:.1e}" for v in a)
        b = "-" if not math.isfinite(r["beta"]) else f"{r['beta']:.2f}"
        lines.append(f"| {r['method']} | {f(r['pertL2_cfd'])} | {f(r['pertL2_exact'])} | {b} |")
    lines.append(f"\nexact shock angle {math.degrees(beta):.3f} deg")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(args.out, "table.md"), "w", encoding="utf-8").write(table + "\n")
    json.dump(rows, open(os.path.join(args.out, "table.json"), "w"), indent=2)

    # profiles
    x = X[:, 0].cpu().numpy(); y = X[:, 1].cpu().numpy(); ref = Q.cpu().numpy()
    colors = {"old FNN (2025-11-18)": "tab:orange", "old notebook (2025-12-02)": "tab:purple",
              "new capturing (no prior)": "tab:blue", "capture -> detect -> fit": "tab:red",
              "shock-fitting (hand-set topology)": "tab:green"}
    xs = (1.5, 1.9)
    fig, axes = plt.subplots(len(xs), 4, figsize=(17, 7.5), constrained_layout=True)
    for i, xc in enumerate(xs):
        xn = x[np.argmin(np.abs(x - xc))]
        col = np.abs(x - xn) < 1e-6
        o = np.argsort(y[col]); yy = y[col][o]; rr = ref[col][o]
        y_w = (xn - dom.w_start) * math.tan(dom.theta)
        yl = torch.linspace(y_w, 0.6, 2000, device=dev)
        Xl = torch.stack([torch.full_like(yl, float(xn)), yl], 1)
        Xe = Xl.clone()
        ex = exact_solution(Xe[:, 0], Xe[:, 1], fs, dom.w_start, dom.theta).cpu().numpy()
        for k, n in enumerate(ev.NAMES):
            ax = axes[i, k]
            sel = yy < 0.6
            ax.plot(yl.cpu().numpy(), ex[:, k], color="0.6", lw=4, alpha=0.6, label="exact")
            ax.plot(yy[sel], rr[sel, k], "k-", lw=1.5, label="CFD")
            for name, m in models.items():
                with torch.no_grad():
                    P = m(Xl).cpu().numpy()
                ax.plot(yl.cpu().numpy(), P[:, k], "--", color=colors.get(name), lw=1.4, label=name)
            ax.set_title(f"x = {xn:.3f}: {ev.LABELS[n]}")
            ax.set_xlabel("y"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "profiles_all_methods.png"), dpi=130)
    plt.close(fig)

    # pressure maps for every method
    names = list(models)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.6 * (len(names) + 1), 3.8), constrained_layout=True)
    from matplotlib.tri import Triangulation
    win = dom.plot_window()
    m = (x >= win[0]) & (x <= win[1]) & (y <= win[3])
    tri = Triangulation(x[m], y[m])
    levels = np.linspace(1.0, post[3] * 1.05, 41)
    for ax, (title, data) in zip(axes, [("CFD", ref[m, 3])] + [(n, preds[n].cpu().numpy()[m, 3]) for n in names]):
        c = ax.tricontourf(tri, np.asarray(data, np.float64), levels=levels, cmap="jet", extend="both")
        ax.set_title(f"p: {title}", fontsize=10); ax.set_aspect("equal")
    fig.colorbar(c, ax=axes, shrink=0.8)
    fig.savefig(os.path.join(args.out, "pressure_all_methods.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
