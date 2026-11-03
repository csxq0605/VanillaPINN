"""Shock-capturing PINN for the wedge, regularised by vanishing artificial viscosity.

One network covers the whole flow; nothing about the shock (where, how many, how strong) is
given. The inviscid Euler equations are replaced by
        div(F, G) = nu * Laplace(U)
and nu is decreased geometrically during training (a continuation from a thick, easy
shock to a thin one). The viscous problem is well posed, so the network cannot "cheat"
near the wedge tip the way it can with the inviscid / weighted formulation.

Second-order equations need extra boundary conditions; the OpenFOAM set-up is copied:
    inlet, top : free stream (Dirichlet)
    symmetry   : v = 0, d(u, rho, p)/dy = 0
    wedge      : u.n = 0, d(rho, p, u_t)/dn = 0   (slip wall)
    outlet     : dq/dx = 0

    python capture.py --tag cap
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch

from physics import Freestream, oblique_shock, viscous_euler_residual, flux_scales, exact_solution
from model import WedgePINN
from data import WedgeDomain, load_reference
import evaluate as ev


@dataclass
class CapConfig:
    # flow / geometry
    mach: float = 5.0
    theta_deg: float = 10.0
    w_start: float = 1.0
    x0: float = 0.0
    x1: float = 2.0
    y0: float = 0.0
    y1: float = 2.0
    # network
    width: int = 64
    depth: int = 6
    adaptive: bool = True
    # collocation (focus band: fraction of the wedge length above the wall)
    n_uniform: int = 6000
    n_focus: int = 10000
    focus_height: float = 0.6
    n_rad: int = 8000
    rad_k: float = 1.0
    rad_c: float = 1.0
    resample_every: int = 500
    n_bc: int = 600
    # artificial viscosity (in units of u_inf * L), decays geometrically nu0 -> nu_min
    nu0: float = 1e-2
    nu_min: float = 5e-4
    nu_decay_frac: float = 0.8      # fraction of the Adam iterations used for the decay
    eps_weight: float = 0.0         # optional detached PINNs-WE weight
    # causal training along x (the flow is supersonic in x, so x plays the role of time):
    # slab i gets weight exp(-causal_eps * sum of the mean residuals of the slabs upstream of it)
    causal_eps: float = 0.0         # 0 = off
    n_slabs: int = 64
    # loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_neu: float = 1.0
    # optimisation
    adam_iters: int = 15000
    lr: float = 1e-3
    lr_final: float = 1e-5
    lbfgs_iters: int = 2000
    # alternative: no Adam, viscosity lowered in `n_stages` steps, each solved with a fresh L-BFGS
    stages: int = 0                 # > 0 switches to the staged L-BFGS mode
    stage_iters: int = 1500         # L-BFGS inner iterations per stage
    theta_start_deg: float = -1.0   # >= 0: wall turning angle is raised from this value to theta_deg
                                    # over the stages (continuation from a weak Mach wave); < 0: off
    seed: int = 0
    # io
    case_dir: str = "../vanilla_Pinn/reference/program_25_11_16"
    case_time: str = "0.005"
    rho_ref: float = 1.16838727127
    p_ref: float = 101325.0
    ref_cache: str = "data/reference_cache.npz"
    out_root: str = "./runs"
    tag: str = "capture"
    log_every: int = 500
    plot_every: int = 5000


class CapturePINN:
    def __init__(self, cfg: CapConfig):
        self.cfg = c = cfg
        torch.manual_seed(c.seed); np.random.seed(c.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fs = Freestream(c.mach)
        self.dom = WedgeDomain(c.x0, c.x1, c.y0, c.y1, c.w_start, c.theta_deg)
        self.beta, self.q_post = oblique_shock(c.mach, self.dom.theta)
        self.q_inf = torch.tensor(self.fs.state(), device=self.device)
        self.scales = torch.tensor(flux_scales(self.fs), device=self.device)
        self.L = self.dom.L
        # viscosity is given relative to u_inf * L so that the same numbers work for any case size
        self.nu_unit = self.fs.u * self.L
        self.focus = (c.w_start - 0.1 * self.L, c.x1, c.focus_height * self.L)
        self.model = WedgePINN(self.dom.bounds, self.fs, c.width, c.depth, c.adaptive).to(self.device)

        if c.case_dir:
            self.X_ref, self.Q_ref = load_reference(c.case_dir, c.case_time, c.ref_cache, self.device, c.rho_ref, c.p_ref)
        else:
            gx = torch.linspace(c.x0, c.x1, 400, device=self.device)
            gy = torch.linspace(c.y0, c.y1, 400, device=self.device)
            Xg = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
            self.X_ref = Xg[self.dom.inside(Xg[:, 0], Xg[:, 1])]
            self.Q_ref = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, c.w_start, self.dom.theta, c.y0)
        self.Q_exact = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, c.w_start, self.dom.theta, c.y0)
        g = torch.Generator().manual_seed(1)
        self.eval_idx = torch.randperm(self.X_ref.shape[0], generator=g)[:40000].to(self.device)

        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = os.path.join(c.out_root, stamp + "_" + c.tag)
        os.makedirs(self.out_dir, exist_ok=True)
        json.dump(asdict(c), open(os.path.join(self.out_dir, "config.json"), "w"), indent=2)
        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8")
        self.hist = []
        self.it = 0
        self.nu_override = None
        self.theta_bc = self.dom.theta      # turning angle imposed by the wall condition
        self.X_fixed = torch.cat([self.dom.sample_uniform(c.n_uniform, self.device),
                                  self.dom.sample_box(c.n_focus, self.focus, self.device)])
        self.X_rad = self.dom.sample_box(c.n_rad, self.focus, self.device)
        self.new_bc()

    # ------------------------------------------------------------- schedule
    def nu(self):
        c = self.cfg
        if self.nu_override is not None:
            return self.nu_override * self.nu_unit
        t = min(1.0, self.it / max(1.0, c.nu_decay_frac * c.adam_iters))
        return c.nu0 * (c.nu_min / c.nu0) ** t * self.nu_unit

    # ------------------------------------------------------------- sampling
    def new_bc(self):
        c, d, dev = self.cfg, self.dom, self.device
        self.bc = d.boundary(c.n_bc, dev)
        y = d.wall_y(torch.tensor([c.x1], device=dev)) + (c.y1 - d.wall_y(torch.tensor([c.x1], device=dev))) * torch.rand(c.n_bc, device=dev)
        self.bc["outlet"] = torch.stack([torch.full_like(y, c.x1), y], 1)

    @torch.no_grad()
    def _res_norm(self, X):
        out = []
        for i in range(0, X.shape[0], 16384):
            with torch.enable_grad():
                r, lam = self.pde_terms(X[i:i + 16384])
            out.append((lam * (r ** 2).sum(1)).detach())
        return torch.cat(out)

    def resample(self):
        c = self.cfg
        cand = torch.cat([self.dom.sample_uniform(4 * c.n_rad, self.device),
                          self.dom.sample_box(4 * c.n_rad, self.focus, self.device)])
        w = self._res_norm(cand) ** c.rad_k
        w = torch.nan_to_num(w / (w.mean() + 1e-30), nan=0.0, posinf=0.0) + c.rad_c
        self.X_rad = cand[torch.multinomial(w, c.n_rad, replacement=False)]
        self.new_bc()

    # --------------------------------------------------------------- losses
    def pde_terms(self, X):
        q, dq, d2q = self.model.forward_derivs(X)
        res, div = viscous_euler_residual(q, dq, d2q, self.fs.gamma, self.nu())
        res = res / self.scales
        if self.cfg.eps_weight > 0:
            lam = (1.0 / (self.cfg.eps_weight * self.L * (div.abs() - div) + 1.0)).detach()
        else:
            lam = torch.ones_like(div)
        return res, lam

    def bc_loss(self):
        m, L = self.model, self.L
        th_bc, th_wall = self.theta_bc, self.dom.theta
        sth, cth = math.sin(th_bc), math.cos(th_bc)          # flow direction imposed at the wall
        swl, cwl = math.sin(th_wall), math.cos(th_wall)      # geometric wall normal
        dir_ = ((m(self.bc["inlet"]) - self.q_inf) ** 2).mean() + ((m(self.bc["top"]) - self.q_inf) ** 2).mean()

        Xs = self.bc["symmetry"]; ws = self.cw(Xs)
        q, dq, _ = m.forward_derivs(Xs, second=False)
        l_sym = (ws * q[:, 1] ** 2).mean()
        n_sym = (ws * ((L * dq[1][:, [0, 2, 3]]) ** 2).sum(1)).mean()

        Xw = self.bc["wedge"]; ww = self.cw(Xw)
        q, dq, _ = m.forward_derivs(Xw, second=False)
        l_wall = (ww * (q[:, 0] * sth - q[:, 1] * cth) ** 2).mean()
        dn = -swl * dq[0] + cwl * dq[1]                       # normal derivative of (u, v, rho, p)
        dut = dn[:, 0] * cwl + dn[:, 1] * swl
        n_wall = (ww * ((L * torch.stack([dut, dn[:, 2], dn[:, 3]], 1)) ** 2).sum(1)).mean()

        Xo = self.bc["outlet"]; wo = self.cw(Xo)
        _, dq, _ = m.forward_derivs(Xo, second=False)
        n_out = (wo * ((L * dq[0]) ** 2).sum(1)).mean()
        return dir_ + l_sym + l_wall, n_sym + n_wall + n_out, l_wall

    def slab(self, x):
        c = self.cfg
        return torch.clamp(((x - c.x0) / (c.x1 - c.x0) * c.n_slabs).long(), 0, c.n_slabs - 1)

    def loss(self, X):
        c = self.cfg
        res, lam = self.pde_terms(X)
        r = lam * (res ** 2).sum(1)
        if c.causal_eps > 0:
            idx = self.slab(X[:, 0])
            cnt = torch.zeros(c.n_slabs, device=self.device).index_add_(0, idx, torch.ones_like(r))
            Ls = torch.zeros(c.n_slabs, device=self.device).index_add_(0, idx, r) / cnt.clamp(min=1)
            upstream = torch.cumsum(Ls, 0) - Ls                        # exclusive prefix sum
            self.W = torch.exp(-c.causal_eps * upstream).detach()
            occupied = cnt > 0
            l_pde = (self.W * Ls)[occupied].mean()
        else:
            self.W = None
            l_pde = r.mean()
        l_bc, l_neu, l_wall = self.bc_loss()
        total = c.w_pde * l_pde + c.w_bc * l_bc + c.w_neu * l_neu
        parts = dict(pde=l_pde.item(), bc=l_bc.item(), neu=l_neu.item(), wall=l_wall.item())
        if self.W is not None:
            parts["w_min"] = self.W.min().item()
        return total, parts

    def cw(self, X):
        """Causal weight of boundary points (1 when causal training is off)."""
        if getattr(self, "W", None) is None:
            return torch.ones(X.shape[0], device=self.device)
        return self.W[self.slab(X[:, 0])]

    # -------------------------------------------------------------- logging
    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + "\n"); self.log_file.flush()

    def metrics(self, idx=None):
        X, Q, Qe = self.X_ref, self.Q_ref, self.Q_exact
        if idx is not None:
            X, Q, Qe = X[idx], Q[idx], Qe[idx]
        res, pred = ev.evaluate(self.model, X, Q, self.dom, self.q_inf, self.beta, self.q_post[3], self.device)
        roi = self.dom.roi_mask(X[:, 0], X[:, 1])
        res["pertL2_roi_vs_exact"] = ev.perturbation_l2(pred, Qe, self.q_inf, roi)
        # post-shock state midway between wall and exact shock, 70% along the wedge
        xp = self.cfg.w_start + 0.7 * self.L
        yp = self.cfg.y0 + (xp - self.cfg.w_start) * 0.5 * (math.tan(self.dom.theta) + math.tan(self.beta))
        with torch.no_grad():
            res["post_shock_state"] = self.model(torch.tensor([[xp, yp]], device=self.device))[0].tolist()
        res["post_shock_exact"] = self.q_post
        return res, pred

    def record(self, total, parts, t0, phase):
        m, _ = self.metrics(self.eval_idx)
        th = [t for (_, _, _, t) in m["shock"] if np.isfinite(t)]
        h = dict(it=self.it, phase=phase, loss=total, **parts, nu=self.nu() / self.nu_unit, time=time.time() - t0,
                 pertL2_roi=m["pertL2_roi"].tolist(), pertL2_exact=m["pertL2_roi_vs_exact"].tolist(),
                 beta=m["beta_pinn"], thickness=float(np.mean(th)) / self.L if th else float("nan"))
        self.hist.append(h)
        self.log(f"[{phase} {self.it:6d}] loss {total:.2e} pde {parts['pde']:.2e} bc {parts['bc']:.2e} "
                 f"neu {parts['neu']:.2e} wmin {parts.get('w_min', 1.0):.2e} | nu {h['nu']:.1e} | beta {h['beta']:.2f} thick/L {h['thickness']:.4f} | "
                 "pertL2(CFD) " + " ".join(f"{v:.3f}" for v in h["pertL2_roi"]) +
                 " | pertL2(exact) " + " ".join(f"{v:.3f}" for v in h["pertL2_exact"]) + f" | {h['time']:.0f}s")

    def snapshot(self, name):
        d = os.path.join(self.out_dir, name)
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"model_{name}.pt"))
        pred = ev.predict(self.model, self.X_ref)
        ev.plot_fields(self.X_ref, self.Q_ref, pred, d, self.dom, tag=f"(capturing, {name})")
        ev.plot_cuts(self.model, self.X_ref, self.Q_ref, self.dom, d, device=self.device)

    # ---------------------------------------------------------------- train
    def lbfgs(self, iters, t0, phase, log_every=200):
        self.resample()
        X = torch.cat([self.X_fixed, self.X_rad])
        opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=50, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        st = {}

        def closure():
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss(X)
            total.backward()
            st["t"], st["p"] = total.item(), parts
            return total

        base = self.it
        for k in range(max(1, iters // 50)):
            opt.step(closure)
            self.it = base + 50 * (k + 1)
            if not math.isfinite(st["t"]):
                self.log("L-BFGS diverged"); return False
            if (50 * (k + 1)) % log_every == 0 or k == iters // 50 - 1:
                self.record(st["t"], st["p"], t0, phase)
        return True

    def train_staged(self):
        c = self.cfg
        t0 = time.time()
        self.log(f"device {self.device}, out {self.out_dir} (staged L-BFGS)")
        for k in range(c.stages):
            t = k / max(1, c.stages - 1)
            self.nu_override = c.nu0 * (c.nu_min / c.nu0) ** t
            if c.theta_start_deg >= 0:
                # angle reaches its final value at the middle stage, the rest only sharpens
                ta = min(1.0, k / max(1, (c.stages - 1) // 2))
                self.theta_bc = math.radians(c.theta_start_deg + (c.theta_deg - c.theta_start_deg) * ta)
            self.log(f"--- stage {k + 1}/{c.stages}: nu = {self.nu_override:.2e} * u_inf L, "
                     f"wall turning {math.degrees(self.theta_bc):.2f} deg")
            if not self.lbfgs(c.stage_iters, t0, f"st{k + 1}"):
                break
            if c.plot_every:
                self.snapshot(f"stage_{k + 1}")
        self.finish(t0)
        return self.out_dir

    def train(self):
        c = self.cfg
        if c.stages > 0:
            return self.train_staged()
        t0 = time.time()
        self.log(f"device {self.device}, out {self.out_dir}")
        self.log(f"exact: beta {math.degrees(self.beta):.3f} deg, post-shock (u,v,rho,p) = "
                 + ", ".join(f"{v:.4f}" for v in self.q_post))
        opt = torch.optim.Adam(self.model.parameters(), lr=c.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (c.lr_final / c.lr) ** (1.0 / max(1, c.adam_iters)))
        for self.it in range(1, c.adam_iters + 1):
            if self.it % c.resample_every == 0:
                self.resample()
            X = torch.cat([self.X_fixed, self.X_rad])
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss(X)
            total.backward()
            opt.step(); sched.step()
            if self.it == 1 or self.it % c.log_every == 0:
                self.record(total.item(), parts, t0, "adam")
            if c.plot_every and self.it % c.plot_every == 0:
                self.snapshot(f"adam_{self.it}")

        if c.lbfgs_iters > 0:
            self.resample()
            X = torch.cat([self.X_fixed, self.X_rad])
            opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=50, history_size=50,
                                    tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
            st = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                total, parts = self.loss(X)
                total.backward()
                st["t"], st["p"] = total.item(), parts
                return total

            base = self.it
            for k in range(max(1, c.lbfgs_iters // 50)):
                opt.step(closure)
                self.it = base + 50 * (k + 1)
                if not math.isfinite(st["t"]):
                    self.log("L-BFGS diverged"); break
                if (k + 1) % 4 == 0:
                    self.record(st["t"], st["p"], t0, "lbfgs")
        self.finish(t0)
        return self.out_dir

    def finish(self, t0):
        m, pred = self.metrics()
        self.log("==== final (all reference points) ====")
        self.log(ev.format_metrics(m))
        self.log("pertL2_roi_vs_exact: " + ", ".join(f"{v:.3e}" for v in m["pertL2_roi_vs_exact"]))
        self.log("post-shock state: " + ", ".join(f"{v:.4f}" for v in m["post_shock_state"]) +
                 " | exact " + ", ".join(f"{v:.4f}" for v in m["post_shock_exact"]))
        self.log(f"total time {time.time() - t0:.0f}s")
        self.snapshot("final")
        ev.plot_history([dict(it=h["it"], loss=h["loss"], pde=h["pde"], bc=h["bc"] + h["neu"], pertL2_roi=h["pertL2_exact"])
                         for h in self.hist], self.out_dir)
        out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}
        json.dump(out, open(os.path.join(self.out_dir, "metrics.json"), "w"), indent=2, default=float)
        json.dump(self.hist, open(os.path.join(self.out_dir, "history.json"), "w"), default=float)
        self.log_file.close()


def parse():
    ap = argparse.ArgumentParser()
    for f in fields(CapConfig):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}[f.type] if isinstance(f.type, str) else f.type
            ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    return CapConfig(**vars(ap.parse_args()))


if __name__ == "__main__":
    CapturePINN(parse()).train()
