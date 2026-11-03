"""Shock-capturing PINN with a learned local artificial viscosity.

This starts from the recipe of the old vanilla_Pinn code (the only capturing variant that
found the right shock) and exposes each ingredient as an option, so they can be changed
one at a time:
    nu_mode  learned : nu(x) = extra_output^2, penalised by w_nu * mean(nu)   (old code)
             none    : inviscid
    d_mode   grad    : residual / d, d = 1 + k(|u_x| - u_x) + k(|v_y| - v_y), not detached (old code)
             detached: same weight but detached
             none
    points   fluent  : the Fluent mesh nodes used by the old code
             random  : uniform + band above the wedge, resampled
    param    raw     : u = o0, v = o1, default init (old code);  freestream: correction to the free stream
Everything else (Euler residual, BCs, metrics, plots) is shared with the rest of v2.

    python capture_lv.py --tag lv_old
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, fields, replace

import numpy as np
import torch

from physics import Freestream, oblique_shock, viscous_euler_residual, flux_scales, exact_solution, fluxes, conservative_gradient
from model import WedgePINN
from data import WedgeDomain, load_reference
import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class LVConfig:
    mach: float = 5.0
    theta_deg: float = 10.0
    w_start: float = 1.0
    x0: float = 0.0
    x1: float = 2.0
    y0: float = 0.0
    y1: float = 2.0
    length_scale: float = 1.0       # coordinates are divided by this (Euler has no length scale)
    # network
    width: int = 15
    depth: int = 4
    adaptive: bool = False
    param: str = "raw"
    normalize_inputs: bool = False
    init_seed: int = 0
    # physics / loss
    nu_mode: str = "learned"
    w_nu: float = 0.01
    nu_min: float = 0.0             # viscosity floor: nu = nu_min + s^2 (keeps the shock resolvable)
    nu_min_end: float = 0.0         # viscosity continuation: >0 lowers the floor geometrically from nu_min to this
    nu_const: bool = False          #   viscosity is the constant floor only (s^2 unused): the thickness is set from outside
    nu_follow: bool = False         #   scale the learned part s^2 by the same factor (nu = eps*(1 + s^2/nu_min)), so the
                                    #   whole viscosity field vanishes, not only the floor
    nu_every: int = 100             #   in stages of this many L-BFGS steps (optimizer restarted at each stage)
    nu_until: int = 600             #   reaching nu_min_end at this step
    visc_form: str = "laplacian"    # "laplacian": nu * Lap(U) (old code); "conservative": div(nu grad U)
    nu_wall: float = 0.0            # >0: nu *= tanh(d_wall / nu_wall)^2, no artificial viscosity at the wall
    d_mode: str = "grad"
    d_coef: float = 0.12
    normalize_res: bool = False
    w_bc: float = 1.0
    tip_delta: float = 0.0          # >0: corner weight r^2/(r^2+delta^2) (r = distance to the wedge apex, delta in units
                                    #   of L) on the residuals; the exact solution is discontinuous at the apex, where
                                    #   the symmetry and wall conditions contradict each other (Kondratiev-type weight)
    tip_pde: bool = True            #   also weight the PDE residual (not only the boundary conditions)
    top_bc: bool = False            # free-stream Dirichlet on the top boundary
    neumann: bool = False           # also d(rho, p, u_t)/dn = 0 on the wedge and the symmetry line
    w_neu: float = 1.0
    # points
    points: str = "fluent"
    n_random: int = 10000
    focus_frac: float = 0.6
    resample_every: int = 0         # in optimiser steps; 0 = fixed points
    n_bc: int = 750
    n_rad: int = 0                  # extra residual-adaptive points (RAD, Wu et al. 2023) on top of the base set
    rad_every: int = 200            # L-BFGS steps between RAD refreshes (L-BFGS is restarted)
    rad_until: int = 0              # no refresh after this step (0 = refresh to the end); then the points stay fixed
    rad_k: float = 1.0              # density ~ |r|^k / mean(|r|^k) + rad_c
    rad_c: float = 1.0
    rad_pool: int = 100000
    # integral (finite-volume) conservation on random squares: enforces the jump conditions even
    # when a shock is thinner than the point spacing; refreshed together with the RAD points
    w_weak: float = 0.0
    n_box: int = 4000
    h_min: float = 0.005            # box sizes as fractions of the wedge length (log-uniform)
    h_max: float = 0.08
    n_quad: int = 32
    box_adapt: float = 0.0          # fraction of boxes centred where the network's compression max(0, -div u) is large
    h_adapt_min: float = 0.03       # sizes of those boxes (fractions of L): larger than the shock, so they straddle it
    h_adapt_max: float = 0.1
    # optimiser
    opt: str = "lbfgs"              # lbfgs (old: lr 0.1, 20 inner iterations / step) or adam+lbfgs
    steps: int = 800
    lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 20
    line_search: str = ""           # "" = fixed step (old code), "strong_wolfe"
    lr_step: int = 200
    lr_gamma: float = 0.8
    adam_iters: int = 0
    adam_lr: float = 1e-3
    seed: int = 5
    init_from: str = ""             # run dir: start from its model_final.pt (continuation stage)
    # io
    case_dir: str = "../vanilla_Pinn/reference/program_25_11_16"
    case_time: str = "0.005"
    rho_ref: float = 1.16838727127
    p_ref: float = 101325.0
    ref_cache: str = "data/reference_cache.npz"
    out_root: str = "./runs"
    tag: str = "lv"
    log_every: int = 50


class LVCapture:
    def __init__(self, cfg):
        ls = cfg.length_scale
        self.cfg = c = replace(cfg, x0=cfg.x0 / ls, x1=cfg.x1 / ls, y0=cfg.y0 / ls, y1=cfg.y1 / ls,
                               w_start=cfg.w_start / ls)
        torch.manual_seed(c.seed); np.random.seed(c.seed)
        self.device = dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.fs = Freestream(c.mach)
        self.dom = WedgeDomain(c.x0, c.x1, c.y0, c.y1, c.w_start, c.theta_deg)
        self.beta, self.q_post = oblique_shock(c.mach, self.dom.theta)
        self.q_inf = torch.tensor(self.fs.state(), device=dev)
        self.scales = torch.tensor(flux_scales(self.fs), device=dev) if c.normalize_res else torch.ones(4, device=dev)
        torch.manual_seed(c.init_seed)
        self.model = WedgePINN(self.dom.bounds, self.fs, c.width, c.depth, c.adaptive, param=c.param,
                               n_extra=1 if c.nu_mode == "learned" else 0,
                               normalize_inputs=c.normalize_inputs).to(dev)
        if c.init_from:
            self.model.load_state_dict(torch.load(os.path.join(c.init_from, "model_final.pt"),
                                                  weights_only=True, map_location=dev))
        torch.manual_seed(c.seed)
        if c.case_dir:
            self.X_ref, self.Q_ref = load_reference(c.case_dir, c.case_time, c.ref_cache, dev, c.rho_ref, c.p_ref)
            self.X_ref = self.X_ref / ls
            self.Q_exact = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, c.w_start, self.dom.theta, c.y0)
        else:
            # no CFD case (--case_dir=): evaluate against the exact oblique-shock solution on a 400x400 grid
            gx = torch.linspace(c.x0, c.x1, 400, device=dev)
            gy = torch.linspace(c.y0, c.y1, 400, device=dev)
            Xg = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
            self.X_ref = Xg[self.dom.inside(Xg[:, 0], Xg[:, 1])]
            self.Q_exact = exact_solution(self.X_ref[:, 0], self.X_ref[:, 1], self.fs, c.w_start, self.dom.theta, c.y0)
            self.Q_ref = self.Q_exact
        g = torch.Generator().manual_seed(1)
        self.eval_idx = torch.randperm(self.X_ref.shape[0], generator=g)[:40000].to(dev)
        self.out_dir = os.path.join(c.out_root, time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + c.tag)
        os.makedirs(self.out_dir, exist_ok=True)
        json.dump(asdict(cfg), open(os.path.join(self.out_dir, "config.json"), "w"), indent=2)
        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8")
        self.hist = []
        self.nu_floor = c.nu_min
        self.sample()

    def sample(self):
        c, d, dev = self.cfg, self.dom, self.device
        if c.points == "fluent":
            data = np.loadtxt(os.path.join(HERE, "data", "fluent_mesh.dat"), skiprows=1)
            self.X = torch.tensor(data[:, 1:3], dtype=torch.float32, device=dev)
        else:
            nf = int(c.focus_frac * c.n_random)
            L = d.L
            parts = [d.sample_uniform(c.n_random - nf, dev)]
            if nf > 0:
                parts.append(d.sample_box(nf, (c.w_start - 0.1 * L, c.x1, 0.6 * L), dev))
            self.X = torch.cat(parts)
        self.n_base = self.X.shape[0]
        n = c.n_bc
        # old code: equally spaced boundary points
        y = torch.linspace(c.y0, c.y1, n, device=dev)
        self.inlet = torch.stack([torch.full_like(y, c.x0), y], 1)
        xs = torch.linspace(c.x0, c.w_start, n + 1, device=dev)[:-1]
        self.sym = torch.stack([xs, torch.full_like(xs, c.y0)], 1)
        xw = torch.linspace(c.w_start, c.x1, n, device=dev)
        self.wall = torch.stack([xw, d.wall_y(xw)], 1)
        xt = torch.linspace(c.x0, c.x1, n, device=dev)
        self.top = torch.stack([xt, torch.full_like(xt, c.y1)], 1)
        self.w_sym = self.tip_weight(self.sym)
        self.w_wall = self.tip_weight(self.wall)
        self.set_points(self.X)

    def tip_weight(self, X):
        c, d = self.cfg, self.dom
        if c.tip_delta <= 0:
            return torch.ones(X.shape[0], device=X.device)
        r2 = (X[:, 0] - d.w_start) ** 2 + (X[:, 1] - d.y0) ** 2
        dl2 = (c.tip_delta * d.L) ** 2
        return r2 / (r2 + dl2)

    def set_points(self, X):
        self.X = X
        self.w_pde = self.tip_weight(X) if self.cfg.tip_pde else torch.ones(X.shape[0], device=X.device)
        if self.cfg.nu_wall > 0:
            self.wall_damp = self.damp(X)

    def damp(self, X):
        return torch.tanh(self.wall_distance(X) / self.cfg.nu_wall) ** 2

    def damp_grad(self, X):
        """Gradient (2, N) of the wall damping factor tanh(d / nu_wall)^2."""
        d = self.dom
        t = torch.tanh(self.wall_distance(X) / self.cfg.nu_wall)
        g = 2 * t * (1 - t * t) / self.cfg.nu_wall                    # d(damp)/d(distance)
        up = X[:, 0] < d.w_start
        nx = torch.where(up, torch.zeros_like(g), torch.full_like(g, -math.sin(d.theta)))
        ny = torch.where(up, torch.ones_like(g), torch.full_like(g, math.cos(d.theta)))
        g = torch.where(self.wall_distance(X) > 0, g, torch.zeros_like(g))
        return torch.stack([g * nx, g * ny])

    def viscosity(self, X, damp=None):
        """nu (N,) and grad nu (2, N) at X; call right after a forward pass of the model at X."""
        c, m = self.cfg, self.model
        s = m.extra[:, 0]
        if c.nu_const:
            s = s * 0.0
        k = self.nu_floor / c.nu_min if (c.nu_follow and c.nu_min > 0) else 1.0
        nu = self.nu_floor + k * s * s
        gnu = 2 * k * s * m.extra_d[:, :, 0] if getattr(m, "extra_d", None) is not None else None
        if c.nu_wall > 0:
            w = self.damp(X) if damp is None else damp
            if gnu is not None:
                gnu = gnu * w + nu * self.damp_grad(X)
            nu = nu * w
        return nu, gnu

    def residual(self, X, damp=None):
        """Scaled (and d-weighted) Euler residual (N,4) and the viscosity nu (N,) at X."""
        c, m = self.cfg, self.model
        q, dq, d2q = m.forward_derivs(X, second=c.nu_mode == "learned")
        if c.nu_mode == "learned":
            nu, gnu = self.viscosity(X, damp)
        else:
            nu, gnu = torch.zeros(X.shape[0], device=X.device), None
        res, _ = viscous_euler_residual(q, dq, d2q, self.fs.gamma, nu if c.nu_mode == "learned" else 0.0)
        if c.nu_mode == "learned" and c.visc_form == "conservative":
            # div(nu grad U) = nu Lap(U) + grad(nu) . grad(U): the second part keeps the jump conditions exact
            gU = conservative_gradient(q, dq, self.fs.gamma)
            res = res - (gnu[0][:, None] * gU[0] + gnu[1][:, None] * gU[1])
        res = res / self.scales
        if c.d_mode != "none":
            ux, vy = dq[0][:, 0], dq[1][:, 1]
            d = c.d_coef * (ux.abs() - ux) + 1.0 + c.d_coef * (vy.abs() - vy)
            if c.d_mode == "detached":
                d = d.detach()
            res = res / d.unsqueeze(1)
        return res, nu

    def new_boxes(self):
        c, d = self.cfg, self.dom
        na = int(c.box_adapt * c.n_box)
        ctr, h = d.sample_boxes(c.n_box - na, c.h_min * d.L, c.h_max * d.L, (d.x0, d.x1, d.y1 - d.y0), 0.0, self.device)
        if na > 0:
            ca, ha = self.compression_boxes(na)
            ctr, h = torch.cat([ctr, ca]), torch.cat([h, ha])
        n, nq = ctr.shape[0], c.n_quad
        s = (torch.arange(nq, device=self.device) + torch.rand(n, nq, device=self.device)) / nq - 0.5
        self.boxes = (ctr, h, s)

    def weak_loss(self):
        """Mean over squares of |closed integral of (F, G).n ds / h|^2 (inviscid fluxes, scaled)."""
        ctr, h, s = self.boxes
        n, nq = ctr.shape[0], self.cfg.n_quad
        hx = h[:, None]; xc, yc = ctr[:, 0:1], ctr[:, 1:2]
        z = 0 * s
        pts = torch.cat([torch.stack([xc + hx / 2 + z, yc + hx * s], -1),
                         torch.stack([xc - hx / 2 + z, yc + hx * s], -1),
                         torch.stack([xc + hx * s, yc + hx / 2 + z], -1),
                         torch.stack([xc + hx * s, yc - hx / 2 + z], -1)], 1).reshape(-1, 2)
        if self.cfg.visc_form == "conservative" and self.cfg.nu_mode == "learned":
            # total flux (F - nu U_x, G - nu U_y), consistent with div(F, G) = div(nu grad U)
            q, dq, _ = self.model.forward_derivs(pts, second=False)
            nu, _ = self.viscosity(pts)
            gU = conservative_gradient(q, dq, self.fs.gamma)
            F, G = fluxes(q, self.fs.gamma)
            F = F - nu[:, None] * gU[0]
            G = G - nu[:, None] * gU[1]
        else:
            F, G = fluxes(self.model(pts), self.fs.gamma)
        F = F.reshape(n, 4, nq, 4); G = G.reshape(n, 4, nq, 4)
        bal = F[:, 0].mean(1) - F[:, 1].mean(1) + G[:, 2].mean(1) - G[:, 3].mean(1)
        return ((bal / torch.tensor(flux_scales(self.fs), device=self.device)) ** 2).sum(1).mean()

    @torch.no_grad()
    def compression_boxes(self, n):
        """Squares centred at points drawn with probability ~ max(0, -div u) of the current network
        (where the network itself puts a shock; no prior), kept only if fully inside the fluid."""
        c, d, dev = self.cfg, self.dom, self.device
        out_c, out_h, need = [], [], n
        while need > 0:
            pool = d.sample_uniform(c.rad_pool, dev)
            comp = []
            for i in range(0, len(pool), 20000):
                _, dq, _ = self.model.forward_derivs(pool[i:i + 20000], second=False)
                comp.append((-(dq[0][:, 0] + dq[1][:, 1])).clamp(min=0.0))
            w = torch.nan_to_num(torch.cat(comp), nan=0.0, posinf=0.0)
            w = w / (w.mean() + 1e-30) + 1e-3
            ctr = pool[torch.multinomial(w, 3 * need, replacement=True)]
            h = c.h_adapt_min * d.L * (c.h_adapt_max / c.h_adapt_min) ** torch.rand(3 * need, device=dev)
            xr = ctr[:, 0] + h / 2
            ok = ((ctr[:, 0] - h / 2 >= d.x0) & (xr <= d.x1) & (ctr[:, 1] + h / 2 <= d.y1)
                  & (ctr[:, 1] - h / 2 >= d.wall_y(xr)))
            out_c.append(ctr[ok]); out_h.append(h[ok])
            need -= int(ok.sum())
        return torch.cat(out_c)[:n], torch.cat(out_h)[:n]

    @torch.no_grad()
    def rad_points(self):
        c = self.cfg
        pool = self.dom.sample_uniform(c.rad_pool, self.device)
        r = torch.cat([self.residual(pool[i:i + 20000])[0].pow(2).sum(1).sqrt() for i in range(0, len(pool), 20000)])
        w = r ** c.rad_k
        w = torch.nan_to_num(w / (w.mean() + 1e-30), nan=0.0, posinf=0.0) + c.rad_c
        idx = torch.multinomial(w, c.n_rad, replacement=False)
        return pool[idx]

    def wall_distance(self, X):
        """Distance to the lower wall (symmetry line upstream of the corner, wedge surface after it)."""
        d = self.dom
        x, y = X[:, 0], X[:, 1]
        dist = torch.where(x < d.w_start, y - d.y0, (y - d.wall_y(x)) * math.cos(d.theta))
        return dist.clamp(min=0.0)

    def loss(self):
        c, m = self.cfg, self.model
        res, nu = self.residual(self.X, self.wall_damp if c.nu_wall > 0 else None)
        l_pde = ((res ** 2) * self.w_pde[:, None]).mean(0).sum()
        l_nu = nu[:self.n_base].mean() if c.nu_mode == "learned" else torch.zeros((), device=self.device)
        th = self.dom.theta
        l_in = ((m(self.inlet) - self.q_inf) ** 2).mean(0).sum()
        l_sym = (self.w_sym * m(self.sym)[:, 1] ** 2).mean()
        qw = m(self.wall)
        l_w = (self.w_wall * (qw[:, 0] * math.sin(th) - qw[:, 1] * math.cos(th)) ** 2).mean()
        l_bc = l_in + l_sym + l_w
        if c.top_bc:
            l_bc = l_bc + ((m(self.top) - self.q_inf) ** 2).mean(0).sum()
        if c.neumann:
            _, dqw, _ = m.forward_derivs(self.wall, second=False)
            nx, ny = math.sin(th), -math.cos(th)
            dn = dqw[0] * nx + dqw[1] * ny                          # (N,4) normal derivatives
            dut = dn[:, 0] * math.cos(th) + dn[:, 1] * math.sin(th)
            l_neu = (dut ** 2).mean() + (dn[:, 2] ** 2).mean() + (dn[:, 3] ** 2).mean()
            _, dqs, _ = m.forward_derivs(self.sym, second=False)
            l_neu = l_neu + (dqs[1][:, [0, 2, 3]] ** 2).mean(0).sum()
            l_bc = l_bc + c.w_neu * l_neu
        total = l_pde + c.w_bc * l_bc + c.w_nu * l_nu
        parts = dict(pde=l_pde.item(), bc=l_bc.item(), nu=l_nu.item())
        if c.w_weak > 0:
            l_weak = self.weak_loss()
            total = total + c.w_weak * l_weak
            parts["weak"] = l_weak.item()
        return total, parts

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
        xp = self.cfg.w_start + 0.7 * self.dom.L
        yp = self.cfg.y0 + (xp - self.cfg.w_start) * 0.5 * (math.tan(self.dom.theta) + math.tan(self.beta))
        with torch.no_grad():
            res["post_shock_state"] = self.model(torch.tensor([[xp, yp]], device=self.device))[0].tolist()
        res["post_shock_exact"] = self.q_post
        return res, pred

    def record(self, step, total, parts, t0):
        m, _ = self.metrics(self.eval_idx)
        th = [t for (_, _, _, t) in m["shock"] if np.isfinite(t)]
        h = dict(it=step, loss=total, **parts, time=time.time() - t0, pertL2_roi=m["pertL2_roi"].tolist(),
                 pertL2_exact=m["pertL2_roi_vs_exact"].tolist(), beta=m["beta_pinn"],
                 thickness=float(np.mean(th)) / self.dom.L if th else float("nan"))
        self.hist.append(h)
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "model_last.pt"))
        self.log(f"[{step:6d}] loss {total:.3e} pde {parts['pde']:.2e} bc {parts['bc']:.2e} nu {parts['nu']:.2e}"
                 + (f" weak {parts['weak']:.2e}" if "weak" in parts else "") + " | "
                 f"beta {h['beta']:.2f} thick/L {h['thickness']:.4f} | pertL2(exact) "
                 + " ".join(f"{v:.3f}" for v in h["pertL2_exact"]) + f" | {h['time']:.0f}s")

    def train(self):
        c = self.cfg
        t0 = time.time()
        self.log(f"device {self.device}, out {self.out_dir}, {self.X.shape[0]} collocation points")
        if c.w_weak > 0:
            self.new_boxes()
        if c.n_rad > 0:
            self.set_points(torch.cat([self.X[:self.n_base], self.rad_points()]))
            self.log(f"RAD: {self.n_base} base + {c.n_rad} adaptive points")
        if c.adam_iters > 0:
            opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
            for it in range(1, c.adam_iters + 1):
                opt.zero_grad(set_to_none=True)
                total, parts = self.loss()
                total.backward()
                opt.step()
                if it % 500 == 0:
                    self.record(-it, total.item(), parts, t0)
            if c.w_weak > 0 and c.box_adapt > 0:
                self.new_boxes()       # adaptive boxes from the warmed-up network, not the random one
        def new_opt(lr):
            return torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=c.lbfgs_max_iter,
                                     line_search_fn=c.line_search or None)

        lr_scale = [1.0]           # halved whenever a step diverges
        lr_at = lambda step: lr_scale[0] * c.lbfgs_lr * c.lr_gamma ** ((step - 1) // c.lr_step)   # StepLR
        import copy
        good = copy.deepcopy(self.model.state_dict())
        opt = new_opt(lr_at(1))
        st = {}

        def closure():
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss()
            total.backward()
            st["t"], st["p"] = total.item(), parts
            return total

        n_stage = max(1, c.nu_until // c.nu_every)
        for step in range(1, c.steps + 1):
            if c.resample_every and step % c.resample_every == 0:
                self.sample()
            if c.nu_min_end > 0 and step > 1 and step % c.nu_every == 1 and step <= c.nu_until + 1:
                k = min((step - 1) // c.nu_every, n_stage)
                self.nu_floor = c.nu_min * (c.nu_min_end / c.nu_min) ** (k / n_stage)
                self.log(f"step {step}: viscosity floor -> {self.nu_floor:.2e}")
                opt = new_opt(lr_at(step))
            if ((c.n_rad > 0 or c.w_weak > 0) and step > 1 and step % c.rad_every == 1
                    and (c.rad_until == 0 or step <= c.rad_until)):
                if c.n_rad > 0:
                    self.set_points(torch.cat([self.X[:self.n_base], self.rad_points()]))
                if c.w_weak > 0:
                    self.new_boxes()
                opt = new_opt(lr_at(step))
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step(closure)
            bad = not math.isfinite(st["t"]) or any(not torch.isfinite(p).all() for p in self.model.parameters())
            if bad:
                if lr_scale[0] < 1e-3:
                    self.log("diverged"); break
                lr_scale[0] *= 0.5
                self.model.load_state_dict(good)
                opt = new_opt(lr_at(step))
                self.log(f"step {step}: non-finite loss, restored last good weights, lr x{lr_scale[0]:g}")
                continue
            good = copy.deepcopy(self.model.state_dict())
            if step % c.log_every == 0 or step == 1:
                self.record(step, st["t"], st["p"], t0)
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
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "model_final.pt"))
        d = os.path.join(self.out_dir, "final")
        ev.plot_fields(self.X_ref, self.Q_ref, pred, d, self.dom, tag=f"(capturing, {self.cfg.tag})")
        ev.plot_cuts(self.model, self.X_ref, self.Q_ref, self.dom, d, device=self.device)
        out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}
        json.dump(out, open(os.path.join(self.out_dir, "metrics.json"), "w"), indent=2, default=float)
        json.dump(self.hist, open(os.path.join(self.out_dir, "history.json"), "w"), default=float)
        self.log_file.close()


def parse():
    ap = argparse.ArgumentParser()
    for f in fields(LVConfig):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}[f.type] if isinstance(f.type, str) else f.type
            ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    return LVConfig(**vars(ap.parse_args()))


if __name__ == "__main__":
    LVCapture(parse()).train()
