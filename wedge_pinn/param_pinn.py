"""Parametric PINN for supersonic flow over a wedge: one network for a range of Mach numbers and wedge
angles. Inputs (x, y, Ma, theta), outputs (u, v, rho, p).

Only the equations are used: steady Euler with artificial viscosity, boundary conditions, a corner weight at
the wedge apex (geometry only) and residual-based collocation points. No shock position, no jump conditions.

Units as in the single-case code: rho_inf = 1, p_inf = 1, velocity / sqrt(p_inf / rho_inf), so
u_inf = sqrt(gamma) Ma. Apex at the origin, wedge length L = 1. Residuals and velocity boundary terms are
divided by their free-stream scale relative to Ma_ref = 5, and the viscosity is scaled with u_inf (fixed
"Reynolds number"), so at Ma 5 the loss is the single-case loss.

    # 1) parameter-modulated first layer, constant viscosity, from scratch (Adam warm-up + fixed-step L-BFGS)
    E:/Conda/envs/cv/python.exe param_pinn.py --arch hyper --nu_mode const --nu 1e-2 --steps 250 --tag pE
    # 2) L-BFGS with strong Wolfe line search
    E:/Conda/envs/cv/python.exe param_pinn.py --init_from runs_param/<pE> --arch hyper --nu_mode const --nu 1e-2 --adam_iters 0 --line_search strong_wolfe --lbfgs_lr 1 --lr_gamma 1 --steps 750 --tag pF
    # 3) boundary weight 10 (keeps the wall from under-turning strong shocks)
    E:/Conda/envs/cv/python.exe param_pinn.py --init_from runs_param/<pF> --arch hyper --nu_mode const --nu 1e-2 --adam_iters 0 --line_search strong_wolfe --lbfgs_lr 1 --lr_gamma 1 --w_bc 10 --steps 500 --tag pG
    # evaluation only
    E:/Conda/envs/cv/python.exe param_pinn.py --init_from runs_param/<run> --arch hyper --steps 0 --adam_iters 0 --tag eval
"""
import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn

from physics import GAMMA, conservative_gradient, fluxes, oblique_shock, viscous_euler_residual

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = GAMMA / (GAMMA - 1.0)


@dataclass
class PConfig:
    # parameter range
    ma_min: float = 3.0
    ma_max: float = 6.0
    th_min: float = 5.0             # wedge angle, degrees
    th_max: float = 20.0
    ma_ref: float = 5.0             # loss scales are relative to this Mach number
    # domain (apex at the origin, wedge length 1); the shock must leave through the outlet x = x1
    x0: float = -0.25
    x1: float = 1.0
    y1: float = 0.9
    # network
    width: int = 128
    depth: int = 6
    coords: str = "cart"            # network input: cart (x, y), polar (r, phi about the apex), both
    r_min: float = 0.002            # polar inputs: no collocation points closer than this to the apex
    arch: str = "mlp"               # mlp: plain MLP on all inputs; hyper: first-layer weights and biases are
                                    # functions of (Ma, theta) (parameter-modulated first layer, see ParamPINN)
    hyper_width: int = 64
    # physics
    nu_mode: str = "learned"        # learned: nu = s(u_inf)*(nu + s^2); const: nu = s(u_inf)*nu
    nu: float = 5e-3                # viscosity (floor) at ma_ref, scaled by u_inf(Ma) / u_inf(ma_ref)
    w_nu: float = 0.01
    eq_scale: str = "ref"           # ref: residuals scaled only for the Ma dependence (old; energy eq ~440x mass at Ma 5);
                                    # own: each conservation law divided by its own free-stream flux (times the Ma_ref
                                    # mass flux), so all four count by relative error
    tip_delta: float = 0.03         # apex corner weight r^2 / (r^2 + delta^2) on PDE and wall / symmetry terms
    w_bc: float = 1.0
    alm: bool = False               # wall no-penetration by augmented Lagrangian on a fixed wall point set
                                    # (PECANN, Basir & Senocak 2022): penalty w_bc plus a multiplier per point,
                                    # updated at every L-BFGS restart, so the wall condition is met without w_bc -> inf
    n_alm: int = 4096
    mirror: bool = False            # slip wall and symmetry line as mirrors: also d(ln rho)/dn = d(ln p)/dn =
                                    # d(u_t)/dn = 0, so the artificial viscosity carries no mass, momentum or
                                    # energy through them (the viscous equations are second order and need these)
    mirror_len: float = 0.1         # length used to make the normal gradients dimensionless (change over this length)
    # integral (control-volume) form: for random squares, the net flux (F - nu U_x, G - nu U_y).n through the
    # boundary (Gauss-Legendre on each side) must vanish -- the same viscous Euler equations integrated over the
    # square (divergence theorem), i.e. a finite-volume flux balance; measures conservation across a shock directly
    w_weak: float = 0.0
    n_box: int = 1024
    n_quad: int = 10               # Gauss points per side
    h_min: float = 0.03            # square side range (wedge length 1); larger than the shock thickness
    h_max: float = 0.06
    box_adapt: float = 0.5         # fraction of squares centred on the residual-sampled (RAD) points
    # points
    n_pts: int = 25000              # uniform in (x, y, Ma, theta)
    n_rad: int = 15000              # residual-based (Wu et al. 2023), density ~ |r|^k / mean + c
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_pool: int = 100000
    n_bc: int = 2000                # per boundary
    resample_every: int = 100       # L-BFGS steps between resampling (L-BFGS restarted)
    stall_resample: bool = True     # also resample when an L-BFGS step leaves the loss unchanged (float32 stall)
    # optimiser
    adam_iters: int = 1000
    adam_lr: float = 1e-3
    steps: int = 2000
    lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 20
    line_search: str = ""           # "" = fixed step lbfgs_lr; "strong_wolfe" = L-BFGS with line search (use lr 1)
    lr_step: int = 400
    lr_gamma: float = 0.8
    # bookkeeping
    log_every: int = 50
    eval_every: int = 250
    snap_every: int = 0             # >0: also save weight snapshots model_s<step>.pt every snap_every steps in the
                                    # second half of training (for iterate averaging; resampling makes iterates jitter)
    seed: int = 0
    init_from: str = ""
    tag: str = "param"
    out_root: str = os.path.join(HERE, "runs_param")


def u_inf(ma):
    return math.sqrt(GAMMA) * ma


# ------------------------------------------------------------------ model
class ParamPINN(nn.Module):
    """MLP (x, y, Ma, theta) -> (u, v, rho, p) + one extra output (local viscosity).
    u = u_inf(Ma) (1 + o0), v = u_inf(Ma) o1, rho = exp(o2), p = exp(o3).

    coords = "polar" feeds the network (r, phi, Ma, theta) with r, phi measured from the apex. The wedge wall
    (phi = theta) and the symmetry line (phi = pi) are then planes in input space for every theta, and any
    ray from the apex is phi = const. The equations are still written and differentiated in (x, y)."""

    def __init__(self, cfg):
        super().__init__()
        self.coords = cfg.coords
        r_max = math.hypot(max(abs(cfg.x0), abs(cfg.x1)), cfg.y1)
        geo_lo = {"cart": [cfg.x0, 0.0], "polar": [0.0, 0.0], "both": [cfg.x0, 0.0, 0.0, 0.0]}[cfg.coords]
        geo_hi = {"cart": [cfg.x1, cfg.y1], "polar": [r_max, math.pi],
                  "both": [cfg.x1, cfg.y1, r_max, math.pi]}[cfg.coords]
        lo = geo_lo + [cfg.ma_min, math.radians(cfg.th_min)]
        hi = geo_hi + [cfg.ma_max, math.radians(cfg.th_max)]
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.arch, self.n_geo = cfg.arch, len(lo) - 2
        layers, d_in, n_plain = [], len(lo), cfg.depth
        if cfg.arch == "hyper":
            # first layer tanh(W(mu) z_geo + b(mu)), W = W0 + dW(mu), b = b0 + db(mu); dW, db from a small MLP of
            # mu = (Ma, theta), zero at start. The parameters can then turn and move every first-layer neuron in
            # (x, y), which is what a parameter-dependent discontinuity needs (shift-DeepONet, Lanthaler et al.
            # 2022; HyperPINN, de Avila Belbute-Peres et al. 2021). Later layers also get mu.
            self.first = nn.Linear(self.n_geo, cfg.width)
            hw = cfg.hyper_width
            self.hyper = nn.Sequential(nn.Linear(2, hw), nn.Tanh(), nn.Linear(hw, hw), nn.Tanh(),
                                       nn.Linear(hw, cfg.width * (self.n_geo + 1)))
            nn.init.zeros_(self.hyper[-1].weight)
            nn.init.zeros_(self.hyper[-1].bias)
            d_in, n_plain = cfg.width + 2, cfg.depth - 1
        for _ in range(n_plain):
            layers += [nn.Linear(d_in, cfg.width), nn.Tanh()]
            d_in = cfg.width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.width, 5)
        self.extra = None

    @staticmethod
    def _us(X):
        return math.sqrt(GAMMA) * X[:, 2]

    def _states(self, o, us):
        return us * (1 + o[:, 0]), us * o[:, 1], torch.exp(o[:, 2]), torch.exp(o[:, 3])

    def features(self, X, derivs=False):
        """Network inputs F and, if asked, dF/dx, dF/dy (2, N, d) and d2F/dx2, d2F/dy2 (2, N, d)."""
        x, y, N = X[:, 0], X[:, 1], X.shape[0]
        cols, d1, d2 = [], [], []
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        if self.coords in ("cart", "both"):
            cols += [x, y]
            d1 += [(one, zero), (zero, one)]
            d2 += [(zero, zero), (zero, zero)]
        if self.coords in ("polar", "both"):
            r2 = (x * x + y * y).clamp(min=1e-12)
            r = r2.sqrt()
            cols += [r, torch.atan2(y, x)]
            d1 += [(x / r, y / r), (-y / r2, x / r2)]
            d2 += [(y * y / (r2 * r), x * x / (r2 * r)), (2 * x * y / (r2 * r2), -2 * x * y / (r2 * r2))]
        F = torch.stack(cols + [X[:, 2], X[:, 3]], 1)
        if not derivs:
            return F
        dF = torch.zeros(2, N, F.shape[1], device=X.device, dtype=X.dtype)
        d2F = torch.zeros_like(dF)
        for j, ((ax, ay), (bx, by)) in enumerate(zip(d1, d2)):
            dF[0, :, j], dF[1, :, j], d2F[0, :, j], d2F[1, :, j] = ax, ay, bx, by
        return F, dF, d2F

    def _first(self, z):
        """Parameter-modulated first layer: pre-activation a (N, width) and the weights W (N, width, n_geo)."""
        g, mu = z[:, :self.n_geo], z[:, self.n_geo:]
        H = self.hyper(mu).view(z.shape[0], -1, self.n_geo + 1)
        W = self.first.weight[None] + H[..., :self.n_geo]
        a = torch.einsum("nij,nj->ni", W, g) + self.first.bias[None] + H[..., self.n_geo]
        return a, W, mu

    def forward(self, X):
        z = 2.0 * (self.features(X) - self.lo) / (self.hi - self.lo) - 1.0
        if self.arch == "hyper":
            a, _, mu = self._first(z)
            z = torch.cat([torch.tanh(a), mu], 1)
        o = self.head(self.body(z))
        self.extra = o[:, 4:]
        return torch.stack(self._states(o, self._us(X)), 1)

    def forward_derivs(self, X, second=True):
        """q (N,4), dq (2,N,4) = (q_x, q_y), d2q (2,N,4) = (q_xx, q_yy) by forward-mode Taylor sweep
        (d2q is None when second=False)."""
        sc = 2.0 / (self.hi - self.lo)
        F, dF, d2F = self.features(X, derivs=True)
        z = sc * (F - self.lo) - 1.0
        h, dh, d2h = z, sc * dF, sc * d2F
        if self.arch == "hyper":
            a, W, mu = self._first(z)          # mu does not depend on (x, y)
            ng = self.n_geo
            da = torch.einsum("nij,knj->kni", W, dh[..., :ng])
            t = torch.tanh(a)
            gt = 1.0 - t * t
            zero = torch.zeros(2, z.shape[0], 2, device=z.device, dtype=z.dtype)
            h = torch.cat([t, mu], 1)
            if second:
                d2a = torch.einsum("nij,knj->kni", W, d2h[..., :ng])
                d2h = torch.cat([gt * d2a - 2.0 * t * gt * da * da, zero], -1)
            dh = torch.cat([gt * da, zero], -1)
        mods = list(self.body)
        for i in range(0, len(mods), 2):
            lin = mods[i]
            a = lin(h)
            da = dh @ lin.weight.T
            h = torch.tanh(a)
            g = 1.0 - h * h
            if second:
                d2h = g * (d2h @ lin.weight.T) - 2.0 * h * g * da * da
            dh = g * da
        o = self.head(h)
        do = dh @ self.head.weight.T
        self.extra = o[:, 4:]
        us = self._us(X)
        u, v, r, p = self._states(o, us)
        q = torch.stack([u, v, r, p], 1)
        dq = torch.stack([us * do[..., 0], us * do[..., 1], r * do[..., 2], p * do[..., 3]], -1)
        if not second:
            return q, dq, None
        d2o = d2h @ self.head.weight.T
        d2q = torch.stack([us * d2o[..., 0], us * d2o[..., 1],
                           r * (d2o[..., 2] + do[..., 2] ** 2), p * (d2o[..., 3] + do[..., 3] ** 2)], -1)
        return q, dq, d2q


# ------------------------------------------------------------------ trainer
class ParamTrainer:
    def __init__(self, cfg):
        self.cfg = c = cfg
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ParamPINN(c).to(self.dev)
        if c.init_from:
            sd = torch.load(os.path.join(c.init_from, "model_final.pt"), map_location=self.dev, weights_only=True)
            self.model.load_state_dict(sd)
        self.out_dir = os.path.join(c.out_root, time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + c.tag)
        os.makedirs(self.out_dir, exist_ok=True)
        json.dump(asdict(c), open(os.path.join(self.out_dir, "config.json"), "w"), indent=2)
        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8")
        self.us_ref = u_inf(c.ma_ref)
        gx, gw = np.polynomial.legendre.leggauss(c.n_quad)     # on [-1, 1]; side coordinate s in [-1/2, 1/2]
        self.gauss = (torch.tensor(gx / 2, dtype=torch.float32, device=self.dev),
                      torch.tensor(gw / 2, dtype=torch.float32, device=self.dev))
        self.hist = []
        if c.alm:
            g = torch.Generator(device="cpu").manual_seed(c.seed + 12345)
            xw = c.x1 * torch.rand(c.n_alm, generator=g)
            ma = c.ma_min + (c.ma_max - c.ma_min) * torch.rand(c.n_alm, generator=g)
            th = math.radians(c.th_min) + math.radians(c.th_max - c.th_min) * torch.rand(c.n_alm, generator=g)
            self.wall_fixed = torch.stack([xw, xw * torch.tan(th), ma, th], 1).to(self.dev)
            self.w_fixed = self.tip_weight(self.wall_fixed)
            self.lam = torch.zeros(c.n_alm, 4 if c.mirror else 1, device=self.dev)
        self.log(f"device {self.dev}, out {self.out_dir}")

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    # ---------------- sampling
    def params(self, n):
        c = self.cfg
        ma = c.ma_min + (c.ma_max - c.ma_min) * torch.rand(n, device=self.dev)
        th = math.radians(c.th_min) + math.radians(c.th_max - c.th_min) * torch.rand(n, device=self.dev)
        return ma, th

    def interior(self, n):
        c, out, need = self.cfg, [], n
        while need > 0:
            m = int(1.3 * need) + 16
            ma, th = self.params(m)
            x = c.x0 + (c.x1 - c.x0) * torch.rand(m, device=self.dev)
            y = c.y1 * torch.rand(m, device=self.dev)
            ok = y > x.clamp(min=0.0) * torch.tan(th)
            if c.coords != "cart":
                ok &= x * x + y * y > c.r_min ** 2
            out.append(torch.stack([x, y, ma, th], 1)[ok])
            need -= int(ok.sum())
        return torch.cat(out)[:n]

    def boundaries(self):
        c, n, dev = self.cfg, self.cfg.n_bc, self.dev
        r = lambda: torch.rand(n, device=dev)
        ma, th = self.params(n)
        inlet = torch.stack([torch.full((n,), c.x0, device=dev), c.y1 * r(), ma, th], 1)
        ma, th = self.params(n)
        sym = torch.stack([c.x0 * r(), torch.zeros(n, device=dev), ma, th], 1)
        ma, th = self.params(n)
        xw = c.x1 * r()
        wall = torch.stack([xw, xw * torch.tan(th), ma, th], 1)
        ma, th = self.params(n)
        top = torch.stack([c.x0 + (c.x1 - c.x0) * r(), torch.full((n,), c.y1, device=dev), ma, th], 1)
        return dict(inlet=inlet, sym=sym, wall=wall, top=top)

    def tip_weight(self, X):
        if self.cfg.tip_delta <= 0:
            return torch.ones(X.shape[0], device=X.device)
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        return r2 / (r2 + self.cfg.tip_delta ** 2)

    def resample(self, with_rad):
        c = self.cfg
        if with_rad and c.n_rad > 0:
            self.X = torch.cat([self.interior(c.n_pts), self.rad_points()])
        else:
            self.X = self.interior(c.n_pts + c.n_rad)
        self.w_tip = self.tip_weight(self.X)
        self.bc = self.boundaries()
        self.bc_w = {k: self.tip_weight(v) for k, v in self.bc.items()}
        if c.w_weak > 0:
            self.new_boxes(adaptive=with_rad and c.n_rad > 0)

    def box_fits(self, ctr, h, th):
        """Square (centre ctr (N,2), side h) inside the domain and above the wedge wall."""
        c, lo = self.cfg, ctr[:, 1] - h / 2
        return ((ctr[:, 0] - h / 2 >= c.x0) & (ctr[:, 0] + h / 2 <= c.x1) & (ctr[:, 1] + h / 2 <= c.y1)
                & (lo >= 0) & (lo >= (ctr[:, 0] + h / 2).clamp(min=0.0) * torch.tan(th)))

    @torch.no_grad()
    def new_boxes(self, adaptive):
        """n_box squares (xc, yc, Ma, theta, h): a fraction centred on the RAD points (drawn from the current
        residual, so they sit where the network itself has its shock), the rest uniform."""
        c, dev, out = self.cfg, self.dev, []
        side = lambda m: c.h_min + (c.h_max - c.h_min) * torch.rand(m, device=dev)
        if adaptive and c.box_adapt > 0:
            P = self.X[c.n_pts:]
            h = side(P.shape[0])
            ok = self.box_fits(P[:, :2], h, P[:, 3])
            out.append(torch.cat([P[ok], h[ok, None]], 1)[: int(c.box_adapt * c.n_box)])
        need = c.n_box - sum(b.shape[0] for b in out)
        while need > 0:
            m = 2 * need + 64
            ma, th = self.params(m)
            h = side(m)
            x = c.x0 + h / 2 + (c.x1 - c.x0 - h) * torch.rand(m, device=dev)
            y = h / 2 + (c.y1 - h) * torch.rand(m, device=dev)
            ok = self.box_fits(torch.stack([x, y], 1), h, th)
            B = torch.stack([x, y, ma, th, h], 1)[ok][:need]
            out.append(B)
            need -= B.shape[0]
        self.boxes = torch.cat(out)

    @torch.no_grad()
    def rad_points(self):
        c = self.cfg
        pool = self.interior(c.rad_pool)
        r = []
        for i in range(0, len(pool), 20000):
            P = pool[i:i + 20000]
            res, _ = self.residual(P)
            r.append((res.pow(2).sum(1) * self.tip_weight(P)).sqrt())
        w = torch.cat(r) ** c.rad_k
        w = torch.nan_to_num(w / (w.mean() + 1e-30), nan=0.0, posinf=0.0) + c.rad_c
        return pool[torch.multinomial(w, c.n_rad, replacement=False)]

    # ---------------- losses
    def wall_vn(self, X):
        """Wall-normal velocity on wall points, divided by the velocity scale of each point."""
        q, th = self.model(X), X[:, 3]
        return (q[:, 0] * torch.sin(th) - q[:, 1] * torch.cos(th)) / self.vscale(X)

    def mirror_grads(self, X, normal):
        """Dimensionless normal derivatives (N, 3) of ln rho, ln p and the tangential velocity / u_inf.
        normal (N, 2) unit normal; tangential direction = normal rotated by -90 degrees."""
        q, dq, _ = self.model.forward_derivs(X)
        dn = normal[:, 0:1] * dq[0] + normal[:, 1:2] * dq[1]
        tx, ty = normal[:, 1], -normal[:, 0]
        L = self.cfg.mirror_len
        return torch.stack([L * dn[:, 2] / q[:, 2], L * dn[:, 3] / q[:, 3],
                            L * (tx * dn[:, 0] + ty * dn[:, 1]) / (math.sqrt(GAMMA) * X[:, 2])], 1)

    def wall_constraints(self, X):
        """Wall conditions as columns: u_n (scaled) and, with mirror, the three normal gradients."""
        cv = self.wall_vn(X)[:, None]
        if not self.cfg.mirror:
            return cv
        th = X[:, 3]
        nrm = torch.stack([-torch.sin(th), torch.cos(th)], 1)
        return torch.cat([cv, self.mirror_grads(X, nrm)], 1)

    @torch.no_grad()
    def alm_update(self):
        """Multiplier step lam <- lam + 2 w_bc c (the gradient of mean(w (lam c + w_bc c^2)) w.r.t. c is
        w (lam + 2 w_bc c) / N, so this is the new multiplier estimate)."""
        with torch.enable_grad():          # forward_derivs uses plain tensor ops; grad mode is irrelevant
            C = self.wall_constraints(self.wall_fixed)
        self.lam += 2.0 * self.cfg.w_bc * C.detach()
        return C

    def vscale(self, X):
        """u_inf(Ma) / u_inf(ma_ref): velocity scale of each point relative to the reference Mach number."""
        return math.sqrt(GAMMA) * X[:, 2] / self.us_ref

    def flux_scale(self, X):
        """Free-stream magnitude of the mass, momentum and energy fluxes, relative to ma_ref (N, 4)."""
        us, ur = math.sqrt(GAMMA) * X[:, 2], self.us_ref
        mass = us / ur
        mom = (us * us + 1.0) / (ur * ur + 1.0)
        en = (G1 * us + 0.5 * us ** 3) / (G1 * ur + 0.5 * ur ** 3)
        if self.cfg.eq_scale == "own":
            # absolute free-stream fluxes (rho_inf = p_inf = 1) in units of the ma_ref mass flux
            return torch.stack([us, us * us + 1.0, us * us + 1.0, G1 * us + 0.5 * us ** 3], 1) / ur
        return torch.stack([mass, mom, mom, en], 1)

    def residual(self, X):
        c = self.cfg
        q, dq, d2q = self.model.forward_derivs(X)
        vs = self.vscale(X)
        s = self.model.extra[:, 0]
        nu = vs * (c.nu + s * s) if c.nu_mode == "learned" else vs * c.nu
        res, _ = viscous_euler_residual(q, dq, d2q, GAMMA, nu)
        return res / self.flux_scale(X), nu / vs

    def box_balance(self, B):
        """Net outward flux of (F - nu U_x, G - nu U_y) through each square, divided by its side h
        (= h times the mean viscous-Euler residual inside), scaled like the PDE residual: (N, 4)."""
        c, n = self.cfg, B.shape[0]
        s, wq = self.gauss
        nq = s.numel()
        xc, yc, h = B[:, 0:1], B[:, 1:2], B[:, 4:5]
        S, half = h * s[None], (h / 2).expand(n, nq)
        xs = torch.stack([xc + half, xc - half, xc + S, xc + S], 1)          # right, left, top, bottom
        ys = torch.stack([yc + S, yc + S, yc + half, yc - half], 1)
        P = torch.stack([xs, ys, B[:, 2, None, None].expand_as(xs), B[:, 3, None, None].expand_as(xs)], -1)
        P = P.reshape(-1, 4)
        q, dq, _ = self.model.forward_derivs(P, second=False)
        vs = self.vscale(P)
        nu = vs * (c.nu + self.model.extra[:, 0] ** 2) if c.nu_mode == "learned" else vs * c.nu
        F, G = fluxes(q, GAMMA)
        gU = conservative_gradient(q, dq, GAMMA)
        F = (F - nu[:, None] * gU[0]).view(n, 4, nq, 4)
        G = (G - nu[:, None] * gU[1]).view(n, 4, nq, 4)
        w = wq[None, :, None]
        bal = (F[:, 0] * w).sum(1) - (F[:, 1] * w).sum(1) + (G[:, 2] * w).sum(1) - (G[:, 3] * w).sum(1)
        return bal / self.flux_scale(B[:, :4])

    def loss(self):
        c, m = self.cfg, self.model
        res, nu_n = self.residual(self.X)
        l_pde = ((res ** 2) * self.w_tip[:, None]).mean(0).sum()
        l_nu = nu_n.mean() if c.nu_mode == "learned" else torch.zeros((), device=self.dev)

        def free(X):
            q, vs = m(X), self.vscale(X)
            us = math.sqrt(GAMMA) * X[:, 2]
            return (((q[:, 0] - us) / vs) ** 2 + (q[:, 1] / vs) ** 2 + (q[:, 2] - 1) ** 2 + (q[:, 3] - 1) ** 2).mean()

        b = self.bc
        l_in, l_top = free(b["inlet"]), free(b["top"])
        l_sym = (self.bc_w["sym"] * (m(b["sym"])[:, 1] / self.vscale(b["sym"])) ** 2).mean()
        if c.alm:
            C = self.wall_constraints(self.wall_fixed)
            l_w = (self.w_fixed[:, None] * C ** 2).mean(0).sum()
            l_alm = (self.w_fixed[:, None] * self.lam * C).mean(0).sum()
        else:
            C = self.wall_constraints(b["wall"])
            l_w = (self.bc_w["wall"][:, None] * C ** 2).mean(0).sum()
            l_alm = torch.zeros((), device=self.dev)
        if c.mirror:   # symmetry line y = 0: d/dy of ln rho, ln p, u vanish
            nrm = torch.zeros(b["sym"].shape[0], 2, device=self.dev); nrm[:, 1] = 1.0
            l_sym = l_sym + (self.bc_w["sym"][:, None] * self.mirror_grads(b["sym"], nrm) ** 2).mean(0).sum()
        l_bc = l_in + l_top + l_sym + l_w
        total = l_pde + c.w_bc * l_bc + l_alm + c.w_nu * l_nu
        parts = dict(pde=l_pde.item(), bc=l_bc.item(), wall=l_w.item(), nu=float(l_nu))
        if c.w_weak > 0:
            l_weak = (self.box_balance(self.boxes) ** 2).sum(1).mean()
            total = total + c.w_weak * l_weak
            parts["weak"] = l_weak.item()
        return total, parts

    # ---------------- training
    def train(self):
        c, t0 = self.cfg, time.time()
        self.resample(with_rad=False)
        if c.adam_iters > 0:
            opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
            for it in range(1, c.adam_iters + 1):
                opt.zero_grad(set_to_none=True)
                total, parts = self.loss()
                total.backward()
                opt.step()
                if it % 500 == 0:
                    self.record(-it, total.item(), parts, t0)
        lr_scale = [1.0]
        lr_at = lambda step: lr_scale[0] * c.lbfgs_lr * c.lr_gamma ** ((step - 1) // c.lr_step)
        new_opt = lambda lr: torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=c.lbfgs_max_iter,
                                                   line_search_fn=c.line_search or None)
        good = copy.deepcopy(self.model.state_dict())
        opt, st, n_stall = new_opt(lr_at(1)), {}, [0]

        def closure():
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss()
            total.backward()
            st["t"], st["p"] = total.item(), parts
            return total

        for step in range(1, c.steps + 1):
            if step > 1 and step % c.resample_every == 1:
                if c.alm:
                    self.alm_update()
                self.resample(with_rad=True)
                opt = new_opt(lr_at(step))
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step(closure)
            bad = not math.isfinite(st["t"]) or any(not torch.isfinite(p).all() for p in self.model.parameters())
            if bad:
                if lr_scale[0] < 1e-3:
                    self.log("diverged")
                    break
                lr_scale[0] *= 0.5
                self.model.load_state_dict(good)
                opt = new_opt(lr_at(step))
                self.log(f"step {step}: non-finite loss, restored last good weights, lr x{lr_scale[0]:g}")
                continue
            good = copy.deepcopy(self.model.state_dict())
            stalled = c.stall_resample and "prev" in st and abs(st["prev"] - st["t"]) <= 1e-7 * abs(st["t"])
            st["prev"] = st["t"]
            if stalled and step % c.resample_every != 0:
                if c.alm:
                    self.alm_update()
                self.resample(with_rad=True)
                opt = new_opt(lr_at(step))
                st.pop("prev")
                n_stall[0] += 1
            if step % c.log_every == 0 or step == 1:
                if n_stall[0]:
                    self.log(f"  ({n_stall[0]} stall resamples since last log)")
                    n_stall[0] = 0
                self.record(step, st["t"], st["p"], t0, full=step % c.eval_every == 0)
            if step % c.log_every == 0:   # save often: the shared desktop GPU sometimes kills a run (CUDA launch failure)
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, "model_last.pt"))
            if c.snap_every and step % c.snap_every == 0 and step > c.steps // 2:
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"model_s{step:04d}.pt"))
        self.finish(t0)
        return self.out_dir

    def record(self, step, total, parts, t0, full=False):
        msg = (f"[{step:6d}] loss {total:.3e} pde {parts['pde']:.2e} bc {parts['bc']:.2e} wall {parts['wall']:.2e} "
               f"nu {parts['nu']:.2e}" + (f" weak {parts['weak']:.2e}" if "weak" in parts else "")
               + f" | {time.time() - t0:.0f}s")
        if self.cfg.alm:
            C = self.wall_constraints(self.wall_fixed).detach()
            rms = (self.w_fixed[:, None] * C ** 2).mean(0).sqrt().tolist()
            msg += (f" | lam mean {self.lam.abs().mean():.2e} max {self.lam.abs().max():.2e}, wall rms "
                    + " ".join(f"{v:.1e}" for v in rms))
        h = dict(step=step, loss=total, **parts, time=time.time() - t0)
        if full or step == 1:
            ev = evaluate_grid(self.model, self.cfg, self.dev, nx=160, ny=120)
            h.update(summary(ev))
            msg += " | " + summary_line(ev)
        self.hist.append(h)
        self.log(msg)

    def finish(self, t0):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "model_final.pt"))
        ev = evaluate_grid(self.model, self.cfg, self.dev, nx=300, ny=220)
        self.log("==== final evaluation (exact oblique-shock solution) ====")
        self.log(" Ma  theta | beta exact  beta PINN (stations)        | thick/L | rho relL2  p relL2 | post-shock rho, p err")
        for e in ev:
            st = ", ".join(f"{b:.2f}" for b in e["beta_stations"])
            self.log(f"{e['ma']:4.1f} {e['theta']:5.1f} | {e['beta_exact']:7.3f}  {e['beta']:7.3f} ({st}) | {e['thickness']:.4f} | "
                     f"{100 * e['relL2'][2]:6.2f}%  {100 * e['relL2'][3]:6.2f}% | {100 * e['post_err'][2]:+.2f}%, {100 * e['post_err'][3]:+.2f}%")
        self.log(summary_line(ev))
        json.dump(dict(cases=ev, summary=summary(ev), total_time=time.time() - t0),
                  open(os.path.join(self.out_dir, "metrics.json"), "w"), indent=1)
        json.dump(self.hist, open(os.path.join(self.out_dir, "history.json"), "w"))
        try:
            plot_all(self.model, self.cfg, ev, self.dev, self.out_dir)
        except Exception as exc:  # plotting must not lose a finished run
            self.log(f"plotting failed: {exc}")
        self.log(f"total time {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ evaluation
def eval_cases(cfg):
    mas = [m for m in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0) if cfg.ma_min - 1e-9 <= m <= cfg.ma_max + 1e-9]
    ths = [t for t in (5.0, 10.0, 15.0, 20.0, 25.0) if cfg.th_min - 1e-9 <= t <= cfg.th_max + 1e-9]
    return [(m, t) for m in mas for t in ths]


@torch.no_grad()
def predict(model, x, y, ma, th, dev):
    X = torch.stack([x, y, torch.full_like(x, ma), torch.full_like(x, th)], 1).to(dev)
    return torch.cat([model(X[i:i + 50000]) for i in range(0, len(X), 50000)]).cpu()


@torch.no_grad()
def evaluate_case(model, cfg, ma, th_deg, dev, nx=300, ny=220):
    th = math.radians(th_deg)
    beta, post = oblique_shock(ma, th)
    qinf = torch.tensor([u_inf(ma), 0.0, 1.0, 1.0])
    post = torch.tensor(post)
    xg, yg = torch.meshgrid(torch.linspace(cfg.x0, cfg.x1, nx), torch.linspace(0, cfg.y1, ny), indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    ok = y >= x.clamp(min=0) * math.tan(th)
    x, y = x[ok], y[ok]
    q = predict(model, x, y, ma, th, dev)
    inside = (x > 0) & (y < x * math.tan(beta))
    qe = torch.where(inside[:, None], post[None], qinf[None])
    err = q - qe
    relL2 = (err.norm(dim=0) / qe.norm(dim=0)).tolist()
    pert = (err.norm(dim=0) / (qe - qinf).norm(dim=0)).tolist()
    # shock position on vertical stations: where p crosses levels between the wall value and p_inf
    bst, thk = [], []
    for xs in (0.25, 0.5, 0.75, 0.95):
        yy = torch.linspace(xs * math.tan(th), cfg.y1, 800)
        p = predict(model, torch.full_like(yy, xs), yy, ma, th, dev)[:, 3]
        pw = p[: max(2, len(p) // 40)].mean().item()
        if pw < 1.05:
            bst.append(float("nan")); thk.append(float("nan")); continue
        lev = lambda f: (yy[p > 1 + f * (pw - 1)].max().item() if (p > 1 + f * (pw - 1)).any() else float("nan"))
        y50 = lev(0.5)
        b = math.atan2(y50, xs)
        bst.append(math.degrees(b))
        thk.append(abs(lev(0.1) - lev(0.9)) * math.cos(b))
    good = [b for b in bst[1:] if b == b]
    beta_p = float(np.mean(good)) if good else float("nan")
    # post-shock state halfway between wall and exact shock, x in [0.6, 0.9]
    xp = torch.linspace(0.6, 0.9, 30)
    yp = xp * 0.5 * (math.tan(th) + math.tan(beta))
    qp = predict(model, xp, yp, ma, th, dev).mean(0)
    return dict(ma=ma, theta=th_deg, beta_exact=math.degrees(beta), beta=beta_p, beta_stations=bst,
                thickness=float(np.nanmean(thk)) if any(t == t for t in thk) else float("nan"),
                relL2=relL2, pertL2=pert, post=qp.tolist(), post_exact=post.tolist(),
                post_err=((qp - post) / post.abs().clamp(min=1e-6)).tolist(),
                exits_outlet=bool(math.tan(beta) * cfg.x1 < cfg.y1))


def evaluate_grid(model, cfg, dev, nx=300, ny=220):
    return [evaluate_case(model, cfg, m, t, dev, nx, ny) for m, t in eval_cases(cfg)]


def summary(ev):
    db = [abs(e["beta"] - e["beta_exact"]) for e in ev]
    db = [d if d == d else 90.0 for d in db]
    return dict(beta_err_mean=float(np.mean(db)), beta_err_max=float(np.max(db)),
                rho_relL2_mean=float(np.mean([e["relL2"][2] for e in ev])),
                p_relL2_mean=float(np.mean([e["relL2"][3] for e in ev])),
                p_relL2_max=float(np.max([e["relL2"][3] for e in ev])))


def summary_line(ev):
    s = summary(ev)
    return (f"beta err mean {s['beta_err_mean']:.2f} max {s['beta_err_max']:.2f} deg | relL2 mean rho "
            f"{100 * s['rho_relL2_mean']:.1f}% p {100 * s['p_relL2_mean']:.1f}% (max p {100 * s['p_relL2_max']:.1f}%)")


def plot_all(model, cfg, ev, dev, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    mas = sorted({e["ma"] for e in ev}); ths = sorted({e["theta"] for e in ev})
    # error maps over the (Ma, theta) plane
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title, f) in zip(axs, [("beta", "激波角误差 (°)", lambda e: e["beta"] - e["beta_exact"]),
                                         ("rho", "ρ 相对 L2 (%)", lambda e: 100 * e["relL2"][2]),
                                         ("p", "p 相对 L2 (%)", lambda e: 100 * e["relL2"][3])]):
        A = np.full((len(mas), len(ths)), np.nan)
        for e in ev:
            A[mas.index(e["ma"]), ths.index(e["theta"])] = f(e)
        cmap = "RdBu_r" if key == "beta" else "viridis"
        vm = np.nanmax(np.abs(A)) if key == "beta" else None
        im = ax.imshow(A, origin="lower", cmap=cmap, vmin=-vm if vm else None, vmax=vm, aspect="auto")
        for i in range(len(mas)):
            for j in range(len(ths)):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="k" if key == "beta" else "w")
        ax.set_xticks(range(len(ths)), [f"{t:g}°" for t in ths]); ax.set_yticks(range(len(mas)), [f"{m:g}" for m in mas])
        ax.set_xlabel("楔角 θ"); ax.set_ylabel("Ma"); ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "error_map.png"), dpi=130); plt.close(fig)
    # pressure fields with the exact shock line
    fig, axs = plt.subplots(len(mas), len(ths), figsize=(3.2 * len(ths), 2.6 * len(mas)), squeeze=False)
    for e in ev:
        ax = axs[mas.index(e["ma"]), ths.index(e["theta"])]
        th = math.radians(e["theta"])
        xg, yg = torch.meshgrid(torch.linspace(cfg.x0, cfg.x1, 250), torch.linspace(0, cfg.y1, 180), indexing="ij")
        q = predict(model, xg.reshape(-1), yg.reshape(-1), e["ma"], th, dev)[:, 3].reshape(xg.shape)
        q[yg < xg.clamp(min=0) * math.tan(th)] = float("nan")
        ax.pcolormesh(xg, yg, q, shading="auto", cmap="viridis")
        xs = np.array([0, cfg.x1])
        ax.plot(xs, xs * math.tan(math.radians(e["beta_exact"])), "w--", lw=0.8)
        ax.fill_between(xs, 0, xs * math.tan(th), color="0.5")
        ax.set_title(f"Ma {e['ma']:g}, θ {e['theta']:g}°: β {e['beta']:.2f} / {e['beta_exact']:.2f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fields_p.png"), dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    for f in fields(PConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            t = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
            ap.add_argument(f"--{f.name}", type=t, default=f.default)
    cfg = PConfig(**vars(ap.parse_args()))
    ParamTrainer(cfg).train()


if __name__ == "__main__":
    main()
