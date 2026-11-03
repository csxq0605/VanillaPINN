"""Shock-fitting PINN for the wedge.

The attached shock is an explicit, trainable curve y_s(x) = (x - x_tip) * tan(beta(x)) starting
at the wedge tip. Two networks are used:
    upstream   N1 on {x < x_tip} U {y > y_s(x)}
    downstream N2 on {x > x_tip, wall(x) < y < y_s(x)}
Each satisfies the smooth Euler equations in its own region; across the curve the
Rankine-Hugoniot conditions  [F . n] = 0  are imposed, which is what moves the shock.
Nothing about the exact solution is used: beta starts at a guess and the curve may bend.

    python shockfit.py --iters 10000 --tag sf
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn as nn

from physics import Freestream, oblique_shock, euler_residual, flux_scales, fluxes, exact_solution
from model import WedgePINN
from data import WedgeDomain, load_reference
import evaluate as ev


@dataclass
class SFConfig:
    mach: float = 5.0
    theta_deg: float = 10.0
    w_start: float = 1.0
    x0: float = 0.0
    x1: float = 2.0
    y0: float = 0.0
    y1: float = 2.0
    width: int = 64
    depth: int = 5
    beta_init_deg: float = 30.0     # initial guess for the shock angle
    curved: bool = True             # let the shock bend (small MLP on top of beta0)
    n_down: int = 8000
    n_up: int = 6000
    n_shock: int = 800
    n_bc: int = 600
    margin: float = 0.05            # each network is also trained a little past the shock
    w_pde: float = 1.0
    w_rh: float = 10.0
    w_bc: float = 10.0
    iters: int = 10000
    lr: float = 2e-3
    lr_final: float = 2e-5
    lr_shock: float = 5e-3
    lbfgs_iters: int = 1000
    resample_every: int = 200
    seed: int = 0
    case_dir: str = "../vanilla_Pinn/reference/program_25_11_16"
    case_time: str = "0.005"
    rho_ref: float = 1.16838727127   # free-stream density of the CFD case [kg/m^3]
    p_ref: float = 101325.0          # free-stream pressure of the CFD case [Pa]
    ref_cache: str = "data/reference_cache.npz"
    out_root: str = "./runs"
    tag: str = "shockfit"
    log_every: int = 500


class ShockCurve(nn.Module):
    """y_s(x) = (x - x_tip) * tan(beta0 + g(x)), g a small zero-initialised MLP."""

    def __init__(self, x_tip, y_tip, x1, beta_init, curved=True):
        super().__init__()
        self.x_tip, self.y_tip, self.x1 = x_tip, y_tip, x1
        self.beta0 = nn.Parameter(torch.tensor(float(beta_init)))
        self.curved = curved
        if curved:
            self.g = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)

    def beta(self, x):
        b = self.beta0.expand_as(x)
        if self.curved:
            s = (2 * (x - self.x_tip) / (self.x1 - self.x_tip) - 1).unsqueeze(1)
            b = b + self.g(s).squeeze(1)
        return b

    def forward(self, x):
        return self.y_tip + (x - self.x_tip) * torch.tan(self.beta(x))


class ShockFitPINN:
    def __init__(self, cfg: SFConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fs = Freestream(cfg.mach)
        self.dom = WedgeDomain(cfg.x0, cfg.x1, cfg.y0, cfg.y1, cfg.w_start, cfg.theta_deg)
        self.beta_exact, self.q_post = oblique_shock(cfg.mach, self.dom.theta)
        self.q_inf = torch.tensor(self.fs.state(), device=self.device)
        self.scales = torch.tensor(flux_scales(self.fs), device=self.device)

        self.up = WedgePINN(self.dom.bounds, self.fs, cfg.width, cfg.depth).to(self.device)
        self.down = WedgePINN(self.dom.bounds, self.fs, cfg.width, cfg.depth).to(self.device)
        self.shock = ShockCurve(cfg.w_start, cfg.y0, cfg.x1, math.radians(cfg.beta_init_deg), cfg.curved).to(self.device)

        if cfg.case_dir:
            self.X_ref, self.Q_ref = load_reference(cfg.case_dir, cfg.case_time, cfg.ref_cache, self.device,
                                                    cfg.rho_ref, cfg.p_ref)
            self.Q_exact = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, cfg.w_start, self.dom.theta, cfg.y0)
        else:
            # no CFD case: evaluate against the exact oblique-shock solution on a 400x400 grid
            gx = torch.linspace(cfg.x0, cfg.x1, 400, device=self.device)
            gy = torch.linspace(cfg.y0, cfg.y1, 400, device=self.device)
            Xg = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
            self.X_ref = Xg[self.dom.inside(Xg[:, 0], Xg[:, 1])]
            self.Q_exact = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, cfg.w_start, self.dom.theta, cfg.y0)
            self.Q_ref = self.Q_exact
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = os.path.join(cfg.out_root, stamp + "_" + cfg.tag)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8")
        self.hist = []
        # probe point between the wedge and the exact shock, 70% along the wedge (only for reporting)
        xp = cfg.w_start + 0.7 * self.dom.L
        yp = cfg.y0 + (xp - cfg.w_start) * 0.5 * (math.tan(self.dom.theta) + math.tan(self.beta_exact))
        self.probe = torch.tensor([[xp, yp]], device=self.device)
        self.sample()

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self):
        c, d, dev = self.cfg, self.dom, self.device
        # downstream: x in (tip, x1], eta in [0, 1 + margin] between the wall and the shock
        x = c.w_start + (c.x1 - c.w_start) * torch.rand(c.n_down, device=dev) ** 0.8
        eta = (1 + c.margin) * torch.rand(c.n_down, device=dev)
        yw, ys = d.wall_y(x), self.shock(x)
        y = torch.clamp(yw + eta * (ys - yw), max=c.y1)
        self.X_down = torch.stack([x, y], 1)
        # upstream: whole box, keep points above (1 - margin) * shock or ahead of the tip
        X = d.sample_uniform(3 * c.n_up, dev)
        ys = self.shock(X[:, 0])
        keep = (X[:, 0] <= c.w_start) | (X[:, 1] >= d.wall_y(X[:, 0]) + (1 - c.margin) * (ys - d.wall_y(X[:, 0])))
        Xu = X[keep]
        # extra upstream points just above the shock
        xs = c.w_start + (c.x1 - c.w_start) * torch.rand(c.n_up // 3, device=dev)
        ysx = self.shock(xs)
        Xs = torch.stack([xs, torch.clamp(ysx + 0.2 * d.L * torch.rand_like(xs), max=c.y1)], 1)
        self.X_up = torch.cat([Xu[: c.n_up], Xs])
        self.bc = d.boundary(c.n_bc, dev)
        # shock parameter positions (the curve itself stays differentiable)
        self.xs = c.w_start + (c.x1 - c.w_start) * torch.rand(c.n_shock, device=dev)

    # -------------------------------------------------------------- losses
    def pde(self, net, X):
        X = X.clone().requires_grad_(True)
        res, _ = euler_residual(net(X), X, self.fs.gamma)
        return ((res / self.scales) ** 2).sum(1).mean()

    def rh(self):
        xs = self.xs.clone().requires_grad_(True)
        ys = self.shock(xs)
        dydx = torch.autograd.grad(ys.sum(), xs, create_graph=True)[0]
        n = torch.stack([-dydx, torch.ones_like(dydx)], 1)
        n = n / n.norm(dim=1, keepdim=True)
        P = torch.stack([xs, ys], 1)
        F1, G1 = fluxes(self.up(P), self.fs.gamma)
        F2, G2 = fluxes(self.down(P), self.fs.gamma)
        jump = (F1 - F2) * n[:, :1] + (G1 - G2) * n[:, 1:]
        return ((jump / self.scales) ** 2).sum(1).mean()

    def bc_loss(self):
        th = self.dom.theta
        l_in = ((self.up(self.bc["inlet"]) - self.q_inf) ** 2).mean()
        l_top = ((self.up(self.bc["top"]) - self.q_inf) ** 2).mean()
        l_sym = (self.up(self.bc["symmetry"])[:, 1] ** 2).mean()
        q = self.down(self.bc["wedge"])
        l_w = ((q[:, 0] * math.sin(th) - q[:, 1] * math.cos(th)) ** 2).mean()
        return l_in + l_top + l_sym, l_w

    def loss(self):
        c = self.cfg
        l_up = self.pde(self.up, self.X_up)
        l_dn = self.pde(self.down, self.X_down)
        l_rh = self.rh()
        l_bc, l_w = self.bc_loss()
        total = c.w_pde * (l_up + l_dn) + c.w_rh * l_rh + c.w_bc * (l_bc + l_w)
        return total, dict(pde_up=l_up.item(), pde_down=l_dn.item(), rh=l_rh.item(), bc=l_bc.item(), wedge=l_w.item())

    # ---------------------------------------------------------- prediction
    @torch.no_grad()
    def predict(self, X, batch=65536):
        out = []
        for i in range(0, X.shape[0], batch):
            Xb = X[i:i + batch]
            down = (Xb[:, 0] > self.cfg.w_start) & (Xb[:, 1] < self.shock(Xb[:, 0]))
            q = self.up(Xb)
            if down.any():
                q[down] = self.down(Xb[down])
            out.append(q)
        return torch.cat(out)

    def shock_angles(self):
        with torch.no_grad():
            xs = torch.tensor(self.dom.stations(), device=self.device)
            return [math.degrees(math.atan2(float(y) - self.cfg.y0, float(x) - self.cfg.w_start)) for x, y in zip(xs, self.shock(xs))]

    def metrics(self):
        pred = self.predict(self.X_ref)
        x, y = self.X_ref[:, 0], self.X_ref[:, 1]
        roi = self.dom.roi_mask(x, y)
        return {
            "relL2_full": ev.rel_l2(pred, self.Q_ref).tolist(),
            "relL2_roi": ev.rel_l2(pred, self.Q_ref, roi).tolist(),
            "pertL2_roi": ev.perturbation_l2(pred, self.Q_ref, self.q_inf, roi).tolist(),
            "pertL2_roi_vs_exact": ev.perturbation_l2(pred, self.Q_exact, self.q_inf, roi).tolist(),
            "cfd_pertL2_roi_vs_exact": ev.perturbation_l2(self.Q_ref, self.Q_exact, self.q_inf, roi).tolist(),
            "beta_stations": self.shock_angles(),
            "beta_exact": math.degrees(self.beta_exact),
            "post_shock_state": self.down(self.probe).tolist()[0],
            "post_shock_exact": self.q_post,
        }, pred

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + "\n"); self.log_file.flush()

    def record(self, it, total, parts, t0, phase):
        m, _ = self.metrics()
        h = dict(it=it, phase=phase, loss=total, **parts, time=time.time() - t0, **m)
        self.hist.append(h)
        self.log(f"[{phase} {it:6d}] loss {total:.3e} pde {parts['pde_up']:.2e}/{parts['pde_down']:.2e} "
                 f"rh {parts['rh']:.2e} bc {parts['bc']:.2e} wedge {parts['wedge']:.2e} | beta "
                 + "/".join(f"{b:.2f}" for b in m["beta_stations"]) +
                 " | pertL2(CFD) " + " ".join(f"{n}={v:.3f}" for n, v in zip(ev.NAMES, m["pertL2_roi"])) +
                 " | pertL2(exact) " + " ".join(f"{n}={v:.3f}" for n, v in zip(ev.NAMES, m["pertL2_roi_vs_exact"])) +
                 f" | {h['time']:.0f}s")

    # ---------------------------------------------------------------- train
    def params(self):
        return list(self.up.parameters()) + list(self.down.parameters()) + list(self.shock.parameters())

    def train(self):
        c = self.cfg
        t0 = time.time()
        self.log(f"device {self.device}, out {self.out_dir}")
        self.log(f"exact: beta {math.degrees(self.beta_exact):.3f} deg, post-shock (u,v,rho,p) = "
                 + ", ".join(f"{v:.4f}" for v in self.q_post) + f"; initial guess beta = {c.beta_init_deg} deg")
        opt = torch.optim.Adam([
            {"params": list(self.up.parameters()) + list(self.down.parameters()), "lr": c.lr},
            {"params": list(self.shock.parameters()), "lr": c.lr_shock},
        ])
        gamma = (c.lr_final / c.lr) ** (1.0 / max(1, c.iters))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        for it in range(1, c.iters + 1):
            if it % c.resample_every == 0:
                self.sample()
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss()
            total.backward()
            opt.step(); sched.step()
            if it == 1 or it % c.log_every == 0:
                self.record(it, total.item(), parts, t0, "adam")

        if c.lbfgs_iters > 0:
            self.sample()
            opt = torch.optim.LBFGS(self.params(), lr=1.0, max_iter=50, history_size=50,
                                    tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
            st = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                total, parts = self.loss()
                total.backward()
                st["t"], st["p"] = total.item(), parts
                return total

            for k in range(max(1, c.lbfgs_iters // 50)):
                opt.step(closure)
                if not math.isfinite(st["t"]):
                    self.log("L-BFGS diverged"); break
                if (k + 1) % 4 == 0:
                    self.record(c.iters + 50 * (k + 1), st["t"], st["p"], t0, "lbfgs")
        self.finish(t0)
        return self.out_dir

    def finish(self, t0):
        m, pred = self.metrics()
        self.log("==== final (all 160k CFD cells) ====")
        for k, v in m.items():
            self.log(f"{k}: {v}")
        self.log(f"total time {time.time() - t0:.0f}s")
        torch.save({"up": self.up.state_dict(), "down": self.down.state_dict(), "shock": self.shock.state_dict()},
                   os.path.join(self.out_dir, "model_final.pt"))
        d = os.path.join(self.out_dir, "final")
        ev.plot_fields(self.X_ref, self.Q_ref, pred, d, self.dom, tag="(shock-fitting PINN)")
        # cuts use a model-like callable
        ev.plot_cuts(_Composite(self), self.X_ref, self.Q_ref, self.dom, d, device=self.device)
        ev.plot_history([dict(it=h["it"], loss=h["loss"], pde=h["pde_up"] + h["pde_down"], bc=h["bc"] + h["wedge"] + h["rh"],
                              pertL2_roi=h["pertL2_roi"]) for h in self.hist], self.out_dir)
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2, default=float)
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.hist, f, default=float)
        self.log_file.close()


class _Composite:
    def __init__(self, sf):
        self.sf = sf

    def __call__(self, X):
        return self.sf.predict(X)


def parse():
    ap = argparse.ArgumentParser()
    for f in fields(SFConfig):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}[f.type] if isinstance(f.type, str) else f.type
            ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    return SFConfig(**vars(ap.parse_args()))


if __name__ == "__main__":
    ShockFitPINN(parse()).train()
