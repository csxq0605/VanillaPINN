"""Capture -> fit: detect the shock in a shock-capturing PINN result, then sharpen it with
the shock-fitting PINN.

Step 1 (detection, no prior on the shock):
    compression indicator  C = max(0, -div u) on a dense grid,
    C > tau * C_max  ->  connected components = candidate shocks,
    ridge (max C per column) -> fitted curve, upstream end -> shock origin.
    An origin within `snap_tol` of a geometric corner is classified as attached there.
Step 2 (refinement): shock-fitting PINN initialised with the detected angle, and both
    sub-networks warm-started from the capturing network.

    python refine.py --capture runs/<capture run> --tag refine
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import Freestream
from model import WedgePINN
from data import WedgeDomain
from shockfit import SFConfig, ShockFitPINN

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_capture(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    fs = Freestream(cfg["mach"])
    ls = cfg.get("length_scale", 1.0)          # the capture net works in coordinates / ls
    dom = WedgeDomain(cfg["x0"] / ls, cfg["x1"] / ls, cfg["y0"] / ls, cfg["y1"] / ls, cfg["w_start"] / ls,
                      cfg["theta_deg"])
    n_extra = 1 if cfg.get("nu_mode") == "learned" else 0
    m = WedgePINN(dom.bounds, fs, cfg["width"], cfg["depth"], cfg["adaptive"], param=cfg.get("param", "freestream"),
                  n_extra=n_extra, normalize_inputs=cfg.get("normalize_inputs", True)).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model_final.pt"), weights_only=True, map_location=dev))
    return cfg, fs, dom, m


@torch.no_grad()
def compression_field(model, dom, n=500):
    """C = max(0, -div u) and grad(rho) on an n x n grid (NaN outside the fluid)."""
    gx = torch.linspace(dom.x0, dom.x1, n, device=dev)
    gy = torch.linspace(dom.y0, dom.y1, n, device=dev)
    X = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    C = torch.empty(X.shape[0], device=dev)
    G = torch.empty(X.shape[0], 2, device=dev)
    for i in range(0, X.shape[0], 32768):
        _, dq, _ = model.forward_derivs(X[i:i + 32768], second=False)
        C[i:i + 32768] = torch.clamp(-(dq[0][:, 0] + dq[1][:, 1]), min=0)
        G[i:i + 32768] = torch.stack([dq[0][:, 2], dq[1][:, 2]], 1)
    inside = dom.inside(X[:, 0], X[:, 1])
    C[~inside] = float("nan")
    return (gx.cpu().numpy(), gy.cpu().numpy(), C.reshape(n, n).cpu().numpy(),
            G.reshape(n, n, 2).cpu().numpy(), inside.reshape(n, n).cpu().numpy())


def _wall_intersection(dom, a, b):
    """x where the line y = a + b x meets the lower wall (symmetry line, then wedge)."""
    x = (dom.y0 - a) / b if b != 0 else dom.x0          # on the symmetry line, x <= tip
    if x > dom.w_start:                                 # otherwise on the wedge surface
        t = math.tan(dom.theta)
        x = (a - dom.y0 + dom.w_start * t) / (t - b) if t != b else dom.x1
    return x, float(dom.wall_y(torch.tensor([x]))[0])


def detect_shocks(model, dom, tau=0.25, min_len=0.2, snap_tol=0.03, margin_cells=3):
    """Shock detection without any prior on the shock.

    Ridge points: C is a local maximum along the shock-normal direction grad(rho)/|grad(rho)|
    (non-maximum suppression, as in edge detection), C > tau * C_max, and the point lies at
    least `margin_cells` grid cells inside the fluid. Ridge points are grouped into connected
    curves; curves longer than min_len * L are shocks. Each is fitted by a robust straight
    line, extended back to the wall to find its origin, and called attached if that origin is
    within snap_tol * L of a wall corner.
    """
    gx, gy, C, G, inside = compression_field(model, dom)
    n = len(gx)
    h = (gx[1] - gx[0], gy[1] - gy[0])
    Cz = np.nan_to_num(C, nan=0.0)
    cmax = np.percentile(Cz[inside], 99.9)
    # interior mask: fluid cells far enough from every boundary
    interior = ndimage.binary_erosion(inside, structure=np.ones((3, 3)), iterations=margin_cells)
    # non-maximum suppression along the normal (in grid index units)
    gn = G / (np.linalg.norm(G, axis=-1, keepdims=True) + 1e-12)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    di, dj = gn[..., 0] / h[0] * h[0], gn[..., 1] / h[1] * h[0]        # one grid step along n
    nrm = np.sqrt(di ** 2 + dj ** 2) + 1e-12
    di, dj = di / nrm, dj / nrm
    fwd = ndimage.map_coordinates(Cz, [ii + di, jj + dj], order=1, mode="nearest")
    bwd = ndimage.map_coordinates(Cz, [ii - di, jj - dj], order=1, mode="nearest")
    ridge = interior & (Cz >= fwd) & (Cz >= bwd) & (Cz > tau * cmax)
    lab, nlab = ndimage.label(ridge, structure=np.ones((3, 3)))
    corners = [(dom.w_start, dom.y0)]
    shocks = []
    for k in range(1, nlab + 1):
        pts = np.argwhere(lab == k)
        xs, ys, ws = gx[pts[:, 0]], gy[pts[:, 1]], Cz[pts[:, 0], pts[:, 1]]
        if np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) < min_len * dom.L:
            continue
        keep = np.ones(len(xs), bool)
        for _ in range(5):                               # robust weighted line fit
            A = np.stack([np.ones(keep.sum()), xs[keep]], 1) * np.sqrt(ws[keep])[:, None]
            a, b = np.linalg.lstsq(A, ys[keep] * np.sqrt(ws[keep]), rcond=None)[0]
            r = ys - (a + b * xs)
            mad = np.median(np.abs(r[keep])) + 1e-9
            keep = np.abs(r) < 4 * 1.4826 * mad
        xo, yo = _wall_intersection(dom, a, b)
        dist = [math.hypot(xo - cx, yo - cy) for cx, cy in corners]
        attached = min(dist) <= snap_tol * dom.L
        if attached:
            xo, yo = corners[int(np.argmin(dist))]
            dx, dy = xs[keep] - xo, ys[keep] - yo                       # refit through the corner
            b_o = np.sum(ws[keep] * dx * dy) / np.sum(ws[keep] * dx * dx)
        else:
            b_o = b
        shocks.append(dict(strength=float(ws.sum()), n_points=int(keep.sum()), origin=(float(xo), float(yo)),
                           raw_origin=tuple(float(v) for v in _wall_intersection(dom, a, b)),
                           attached=bool(attached), angle_deg=math.degrees(math.atan(b_o)),
                           max_dev=float(np.abs(r[keep]).max()),
                           ridge=(xs[keep].tolist(), ys[keep].tolist())))
    shocks.sort(key=lambda s: -s["strength"])
    # a thin ridge can break into pieces on the grid: merge pieces with the same origin and angle
    merged = []
    for s in shocks:
        for t in merged:
            if (math.dist(s["origin"], t["origin"]) <= snap_tol * dom.L
                    and abs(s["angle_deg"] - t["angle_deg"]) < 1.0):
                w = s["strength"] / (s["strength"] + t["strength"])
                t["angle_deg"] = (1 - w) * t["angle_deg"] + w * s["angle_deg"]
                t["strength"] += s["strength"]; t["n_points"] += s["n_points"]
                t["max_dev"] = max(t["max_dev"], s["max_dev"])
                t["ridge"] = (t["ridge"][0] + s["ridge"][0], t["ridge"][1] + s["ridge"][1])
                t["pieces"] = t.get("pieces", 1) + 1
                break
        else:
            merged.append(s)
    return merged, (gx, gy, Cz)


def plot_detection(shocks, field, dom, path):
    gx, gy, C = field
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    win = dom.plot_window()
    c = ax.pcolormesh(gx, gy, C.T, cmap="magma", shading="auto", vmax=np.percentile(C, 99.9))
    fig.colorbar(c, ax=ax, label=r"compression $\max(0,-\nabla\cdot u)$")
    for k, s in enumerate(shocks):
        xs, ys = s["ridge"]
        ax.plot(xs, ys, ".", ms=2, color="cyan", label="ridge" if k == 0 else None)
        xo, yo = s["origin"]
        xl = np.linspace(xo, dom.x1, 50)
        ax.plot(xl, yo + (xl - xo) * math.tan(math.radians(s["angle_deg"])), "w--", lw=1,
                label=f"fit {s['angle_deg']:.2f} deg" + (" (attached)" if s["attached"] else ""))
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lbfgs_iters", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warm_start", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--tag", default="refine")
    ap.add_argument("--width", type=int, default=0, help="shock-fitting nets; 0 = same as the capture net")
    ap.add_argument("--depth", type=int, default=0)
    args = ap.parse_args()

    ccfg, fs, dom, cap = load_capture(args.capture)
    shocks, field = detect_shocks(cap, dom, tau=args.tau)
    print(f"detected {len(shocks)} shock(s)")
    for s in shocks:
        print({k: v for k, v in s.items() if k != "ridge"})
    if not shocks:
        raise SystemExit("no shock detected")
    s = shocks[0]
    if len(shocks) > 1:
        print("warning: more than one shock detected; only the strongest is fitted")
    if not s["attached"]:
        raise SystemExit("the detected shock is not attached to a wall corner; this topology is not implemented")

    compatible = (ccfg.get("param", "freestream") == "freestream" and ccfg.get("nu_mode", "none") != "learned"
                  and ccfg.get("normalize_inputs", True) and not args.width and not args.depth)
    width = args.width or (ccfg["width"] if compatible else SFConfig.width)
    depth = args.depth or (ccfg["depth"] if compatible else SFConfig.depth)
    ls = ccfg.get("length_scale", 1.0)
    sfc = SFConfig(mach=ccfg["mach"], theta_deg=ccfg["theta_deg"], w_start=s["origin"][0] * ls,
                   x0=ccfg["x0"], x1=ccfg["x1"], y0=ccfg["y0"], y1=ccfg["y1"],
                   width=width, depth=depth, beta_init_deg=s["angle_deg"],
                   iters=args.iters, lbfgs_iters=args.lbfgs_iters, lr=args.lr,
                   case_dir=ccfg["case_dir"], case_time=ccfg["case_time"], rho_ref=ccfg["rho_ref"],
                   p_ref=ccfg["p_ref"], ref_cache=ccfg["ref_cache"], out_root=ccfg["out_root"], tag=args.tag)
    sf = ShockFitPINN(sfc)
    if args.warm_start and compatible:
        sd = cap.state_dict()
        sf.up.load_state_dict(sd); sf.down.load_state_dict(sd)
    plot_detection(shocks, field, dom, os.path.join(sf.out_dir, "detection.png"))
    json.dump(dict(capture_run=args.capture, shocks=[{k: v for k, v in t.items() if k != "ridge"} for t in shocks]),
              open(os.path.join(sf.out_dir, "detection.json"), "w"), indent=2)
    sf.log(f"capture run: {args.capture} (weights copied: {bool(args.warm_start and compatible)})")
    sf.log(f"detected shock: origin {s['origin']} (attached {s['attached']}, raw {s['raw_origin']}), "
           f"angle {s['angle_deg']:.3f} deg, max deviation from a line {s['max_dev']:.4f}")
    sf.train()


if __name__ == "__main__":
    main()
