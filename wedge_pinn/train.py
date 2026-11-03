"""Training loop: Adam (with residual-based resampling) followed by L-BFGS."""
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from physics import Freestream, oblique_shock, euler_residual, flux_scales, fluxes
from model import WedgePINN
from data import WedgeDomain, load_reference
import evaluate as ev


@dataclass
class Config:
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
    # collocation
    n_uniform: int = 6000        # whole domain
    n_focus: int = 10000         # band above the wedge, where the shock lives
    focus_height: float = 0.6     # band above the wedge, as a fraction of the wedge length
    n_rad: int = 8000            # residual-adaptive points, refreshed every `resample_every`
    rad_k: float = 1.0
    rad_c: float = 1.0
    resample_every: int = 1000
    n_bc: int = 600
    # loss
    eps_weight: float = 0.1      # PINNs-WE weight 1 / (eps (|div u| - div u) + 1); 0 disables
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_wedge: float = 10.0
    use_top_bc: bool = True
    nu0: float = 0.0             # artificial viscosity at start, decays to 0 by nu_end_frac * adam_iters
    nu_end_frac: float = 0.7
    nu_final: float = 0.0        # viscosity kept after the decay (and during L-BFGS)
    # weak-form (finite-volume) flux balance on random squares
    w_weak: float = 0.0
    n_box: int = 4000
    h_min: float = 0.005         # box sizes, as fractions of the wedge length
    h_max: float = 0.08
    n_quad: int = 16
    # optimisation
    adam_iters: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5
    lbfgs_iters: int = 3000
    lbfgs_lr: float = 1.0
    seed: int = 0
    # io
    case_dir: str = "../vanilla_Pinn/reference/program_25_11_16"
    case_time: str = "0.005"
    rho_ref: float = 1.16838727127   # free-stream density of the CFD case [kg/m^3]
    p_ref: float = 101325.0          # free-stream pressure of the CFD case [Pa]
    ref_cache: str = "data/reference_cache.npz"
    out_root: str = "./runs"
    tag: str = ""
    log_every: int = 500
    plot_every: int = 5000


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fs = Freestream(cfg.mach)
        self.dom = WedgeDomain(cfg.x0, cfg.x1, cfg.y0, cfg.y1, cfg.w_start, cfg.theta_deg)
        self.beta, self.q_post = oblique_shock(cfg.mach, self.dom.theta)
        self.q_inf = torch.tensor(self.fs.state(), device=self.device)
        self.scales = torch.tensor(flux_scales(self.fs), device=self.device)
        L = self.dom.L
        self.focus = (cfg.w_start - 0.1 * L, cfg.x1, cfg.focus_height * L)
        self.model = WedgePINN(self.dom.bounds, self.fs, cfg.width, cfg.depth, cfg.adaptive).to(self.device)

        self.X_ref, self.Q_ref = load_reference(cfg.case_dir, cfg.case_time, cfg.ref_cache, self.device, cfg.rho_ref, cfg.p_ref)
        g = torch.Generator(device="cpu").manual_seed(1)
        self.eval_idx = torch.randperm(self.X_ref.shape[0], generator=g)[:40000].to(self.device)

        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = os.path.join(cfg.out_root, stamp + (("_" + cfg.tag) if cfg.tag else ""))
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        self.hist = []
        self.it = 0
        self.weak_fixed = None   # set during L-BFGS so the loss is deterministic
        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8")

        self.X_fixed = torch.cat([
            self.dom.sample_uniform(cfg.n_uniform, self.device),
            self.dom.sample_box(cfg.n_focus, self.focus, self.device),
        ])
        self.X_rad = self.dom.sample_box(cfg.n_rad, self.focus, self.device)
        self.bc = self.dom.boundary(cfg.n_bc, self.device)

    # ------------------------------------------------------------------ losses
    def nu(self):
        c = self.cfg
        if c.nu0 <= 0:
            return c.nu_final
        t = min(1.0, self.it / max(1.0, c.nu_end_frac * c.adam_iters))
        return c.nu_final + (c.nu0 - c.nu_final) * (1.0 - t) ** 2

    def pde_terms(self, X):
        X = X.clone().requires_grad_(True)
        q = self.model(X)
        res, div = euler_residual(q, X, self.fs.gamma, self.nu())
        res = res / self.scales
        if self.cfg.eps_weight > 0:
            # detached: the network must not be able to lower its loss by inventing compressions
            lam = 1.0 / (self.cfg.eps_weight * (div.abs() - div) + 1.0)
            lam = lam.detach()
        else:
            lam = torch.ones_like(div)
        return res, lam

    def bc_loss(self):
        m, fs, th = self.model, self.fs, self.dom.theta
        q_in = m(self.bc["inlet"])
        l_in = ((q_in - self.q_inf) ** 2).mean()
        l_top = ((m(self.bc["top"]) - self.q_inf) ** 2).mean() if self.cfg.use_top_bc else q_in.new_zeros(())
        l_sym = (m(self.bc["symmetry"])[:, 1] ** 2).mean()
        q_w = m(self.bc["wedge"])
        l_w = ((q_w[:, 0] * math.sin(th) - q_w[:, 1] * math.cos(th)) ** 2).mean()
        return l_in, l_top, l_sym, l_w

    def new_boxes(self):
        c = self.cfg
        ctr, h = self.dom.sample_boxes(c.n_box, c.h_min * self.dom.L, c.h_max * self.dom.L,
                                       self.focus, 0.7, self.device)
        n, nq = ctr.shape[0], c.n_quad
        # stratified offsets along each edge, in [-0.5, 0.5]
        s = (torch.arange(nq, device=self.device) + torch.rand(n, nq, device=self.device)) / nq - 0.5
        return ctr, h, s

    def weak_loss(self, boxes=None):
        """Flux balance over random squares: (1/h) * closed integral of (F, G).n ds = 0.
        Edges use stratified random quadrature. Zero for any conservative solution, sharp or not,
        so it enforces the Rankine-Hugoniot jumps without penalising the shock itself."""
        c = self.cfg
        if boxes is None:
            boxes = self.weak_fixed or self.new_boxes()
        ctr, h, s = boxes
        n, nq = ctr.shape[0], c.n_quad
        hx = h[:, None]
        xc, yc = ctr[:, 0:1], ctr[:, 1:2]
        # right, left, top, bottom edges
        pts = torch.cat([
            torch.stack([xc + hx / 2 + 0 * s, yc + hx * s], -1),
            torch.stack([xc - hx / 2 + 0 * s, yc + hx * s], -1),
            torch.stack([xc + hx * s, yc + hx / 2 + 0 * s], -1),
            torch.stack([xc + hx * s, yc - hx / 2 + 0 * s], -1),
        ], 1).reshape(-1, 2)
        F, G = fluxes(self.model(pts), self.fs.gamma)
        F = F.reshape(n, 4, nq, 4); G = G.reshape(n, 4, nq, 4)
        # (sum over the edge) * (h / nq) / h  ==  mean over edge points
        bal = F[:, 0].mean(1) - F[:, 1].mean(1) + G[:, 2].mean(1) - G[:, 3].mean(1)   # (n, 4)
        return ((bal / self.scales) ** 2).sum(1).mean()

    def loss(self, X):
        c = self.cfg
        if c.w_pde > 0:
            res, lam = self.pde_terms(X)
            l_pde = (lam * (res ** 2).sum(1)).mean()
        else:
            l_pde = torch.zeros((), device=self.device)
        l_weak = self.weak_loss() if c.w_weak > 0 else torch.zeros((), device=self.device)
        l_in, l_top, l_sym, l_w = self.bc_loss()
        l_bc = l_in + l_top + l_sym
        total = c.w_pde * l_pde + c.w_weak * l_weak + c.w_bc * l_bc + c.w_wedge * l_w
        return total, {"pde": l_pde.item(), "weak": l_weak.item(), "bc": (l_bc + l_w).item(), "wedge": l_w.item()}

    # ------------------------------------------------------------- resampling
    def resample(self):
        """Residual-based adaptive distribution (Wu et al. 2023): p(x) ~ r^k / mean(r^k) + c."""
        c = self.cfg
        cand = torch.cat([
            self.dom.sample_uniform(4 * c.n_rad, self.device),
            self.dom.sample_box(4 * c.n_rad, self.focus, self.device),
        ])
        r = []
        for i in range(0, cand.shape[0], 16384):
            res, lam = self.pde_terms(cand[i:i + 16384])
            r.append((lam * (res ** 2).sum(1)).detach())
        r = torch.cat(r)
        w = r ** c.rad_k
        w = torch.nan_to_num(w / (w.mean() + 1e-30), nan=0.0, posinf=0.0) + c.rad_c
        idx = torch.multinomial(w, c.n_rad, replacement=False)
        self.X_rad = cand[idx].detach()
        self.bc = self.dom.boundary(c.n_bc, self.device)

    # --------------------------------------------------------------- logging
    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def record(self, total, parts, phase, t0, full=False):
        Xe, Qe = self.X_ref[self.eval_idx], self.Q_ref[self.eval_idx]
        res, _ = ev.evaluate(self.model, Xe, Qe, self.dom, self.q_inf, self.beta, self.q_post[3], self.device)
        h = {"it": self.it, "phase": phase, "loss": total, **parts,
             "relL2_full": res["relL2_full"].tolist(), "pertL2_roi": res["pertL2_roi"].tolist(),
             "beta": res["beta_pinn"], "time": time.time() - t0, "nu": self.nu()}
        self.hist.append(h)
        self.log(f"[{phase} {self.it:6d}] loss {total:.3e} pde {parts['pde']:.3e} weak {parts['weak']:.3e} bc {parts['bc']:.3e} "
                 f"wedge {parts['wedge']:.2e} | pertL2 " +
                 " ".join(f"{n}={v:.3f}" for n, v in zip(ev.NAMES, h["pertL2_roi"])) +
                 f" | beta {h['beta']:.2f} | nu {h['nu']:.1e} | {h['time']:.0f}s")

    def snapshot(self, name):
        d = os.path.join(self.out_dir, name)
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"model_{name}.pt"))
        pred = ev.predict(self.model, self.X_ref)
        ev.plot_fields(self.X_ref, self.Q_ref, pred, d, self.dom, tag=f"({name})")
        ev.plot_cuts(self.model, self.X_ref, self.Q_ref, self.dom, d, device=self.device)

    # ------------------------------------------------------------------ train
    def train(self):
        c = self.cfg
        t0 = time.time()
        self.log(f"device {self.device}, out {self.out_dir}")
        self.log(f"exact: beta {math.degrees(self.beta):.3f} deg, post-shock (u,v,rho,p) = "
                 + ", ".join(f"{v:.4f}" for v in self.q_post))

        opt = torch.optim.Adam(self.model.parameters(), lr=c.lr)
        gamma = (c.lr_final / c.lr) ** (1.0 / max(1, c.adam_iters))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        for self.it in range(1, c.adam_iters + 1):
            if self.it % c.resample_every == 0:
                self.resample()
            X = torch.cat([self.X_fixed, self.X_rad])
            opt.zero_grad(set_to_none=True)
            total, parts = self.loss(X)
            total.backward()
            opt.step()
            sched.step()
            if self.it % c.log_every == 0 or self.it == 1:
                self.record(total.item(), parts, "adam", t0)
            if c.plot_every and self.it % c.plot_every == 0:
                self.snapshot(f"adam_{self.it}")

        if c.lbfgs_iters > 0:
            # viscosity must be off here (nu() reads self.it, so push it past the schedule)
            self.it = c.adam_iters
            self.resample()
            X = torch.cat([self.X_fixed, self.X_rad])
            if c.w_weak > 0:
                self.weak_fixed = self.new_boxes()
            opt = torch.optim.LBFGS(self.model.parameters(), lr=c.lbfgs_lr, max_iter=50, history_size=50,
                                    tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
            state = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                total, parts = self.loss(X)
                total.backward()
                state["total"], state["parts"] = total.item(), parts
                return total

            n_outer = max(1, c.lbfgs_iters // 50)
            for k in range(n_outer):
                opt.step(closure)
                self.it = c.adam_iters + (k + 1) * 50
                if not math.isfinite(state["total"]):
                    self.log("L-BFGS diverged, stopping")
                    break
                if (k + 1) % max(1, c.log_every // 50) == 0 or k == n_outer - 1:
                    self.record(state["total"], state["parts"], "lbfgs", t0)

        self.finish(t0)
        return self.out_dir

    def finish(self, t0):
        res, pred = ev.evaluate(self.model, self.X_ref, self.Q_ref, self.dom, self.q_inf, self.beta,
                                self.q_post[3], self.device)
        self.log("==== final (all 160k CFD cells) ====")
        self.log(ev.format_metrics(res))
        self.log(f"total time {time.time() - t0:.0f}s")
        self.snapshot("final")
        ev.plot_history(self.hist, self.out_dir)
        summary = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.hist, f, default=float)
        self.log_file.close()
