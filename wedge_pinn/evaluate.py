"""Error metrics, shock-angle estimate and figures."""
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

NAMES = ["u", "v", "rho", "p"]
LABELS = {"u": "u", "v": "v", "rho": r"$\rho$", "p": "p"}


@torch.no_grad()
def predict(model, X, batch=65536):
    return torch.cat([model(X[i:i + batch]) for i in range(0, X.shape[0], batch)])


def rel_l2(pred, ref, mask=None):
    if mask is not None:
        pred, ref = pred[mask], ref[mask]
    num = torch.linalg.norm(pred - ref, dim=0)
    den = torch.linalg.norm(ref, dim=0)
    return (num / den).cpu().numpy()


def perturbation_l2(pred, ref, q_inf, mask=None):
    """||pred - ref|| / ||ref - q_inf||: error relative to the size of the flow disturbance.
    Much stricter than plain rel-L2, which is dominated by the uniform free stream."""
    if mask is not None:
        pred, ref = pred[mask], ref[mask]
    num = torch.linalg.norm(pred - ref, dim=0)
    den = torch.linalg.norm(ref - q_inf, dim=0)
    return (num / den).cpu().numpy()


def shock_profile(model, domain, p_post, xs=None, n=4000, device="cpu"):
    """For each x station, locate the shock (p crosses the mid value) and its 10-90% thickness."""
    out = []
    for xc in (xs or domain.stations()):
        y_w = domain.y0 + (xc - domain.w_start) * math.tan(domain.theta)
        y = torch.linspace(y_w, min(domain.y1, y_w + domain.L), n, device=device)
        X = torch.stack([torch.full_like(y, xc), y], 1)
        p = predict(model, X)[:, 3].cpu().numpy()
        y = y.cpu().numpy()
        lo, hi = 1.0, p_post
        def cross(level):
            above = np.where(p > level)[0]
            return y[above.max()] if len(above) else np.nan
        ys = cross(0.5 * (lo + hi))
        thick = cross(lo + 0.1 * (hi - lo)) - cross(lo + 0.9 * (hi - lo))
        ang = math.degrees(math.atan2(ys - domain.y0, xc - domain.w_start)) if np.isfinite(ys) else np.nan
        out.append((xc, ys, ang, thick))
    return out


def evaluate(model, X_ref, Q_ref, domain, q_inf, beta_exact, p_post, device):
    pred = predict(model, X_ref)
    x, y = X_ref[:, 0], X_ref[:, 1]
    roi = domain.roi_mask(x, y)
    res = {
        "relL2_full": rel_l2(pred, Q_ref),
        "relL2_roi": rel_l2(pred, Q_ref, roi),
        "pertL2_roi": perturbation_l2(pred, Q_ref, q_inf, roi),
    }
    prof = shock_profile(model, domain, p_post, device=device)
    res["shock"] = prof
    angles = [a for (_, _, a, _) in prof if np.isfinite(a)]
    res["beta_pinn"] = float(np.mean(angles)) if angles else float("nan")
    res["beta_exact"] = math.degrees(beta_exact)
    return res, pred


def format_metrics(res):
    s = []
    for key in ["relL2_full", "relL2_roi", "pertL2_roi"]:
        s.append(key + ": " + ", ".join(f"{n}={v:.3e}" for n, v in zip(NAMES, res[key])))
    s.append(f"shock angle: PINN {res['beta_pinn']:.2f} deg, exact {res['beta_exact']:.2f} deg")
    s.append("shock stations (x, y_shock, angle, 10-90% thickness): " +
             "; ".join(f"({a:.2f}, {b:.3f}, {c:.2f}, {d:.3f})" for a, b, c, d in res["shock"]))
    return "\n".join(s)


def plot_fields(X, Q_ref, Q_pred, out_dir, domain, tag=""):
    """Side-by-side CFD / PINN / |error| contours in the shock region."""
    os.makedirs(out_dir, exist_ok=True)
    window = domain.plot_window()
    x = X[:, 0].cpu().numpy(); y = X[:, 1].cpu().numpy()
    m = (x >= window[0]) & (x <= window[1]) & (y >= window[2]) & (y <= window[3])
    x, y = x[m], y[m]
    ref = Q_ref.cpu().numpy()[m].astype(np.float64); pred = Q_pred.cpu().numpy()[m].astype(np.float64)
    tri = Triangulation(x, y)
    # drop slivers below the wedge created by triangulating the concave corner
    xc = x[tri.triangles].mean(1); yc = y[tri.triangles].mean(1)
    tri.set_mask(yc < domain.y0 + np.clip(xc - domain.w_start, 0, None) * math.tan(domain.theta) - 1e-6 * domain.L)

    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    for k, n in enumerate(NAMES):
        vmin, vmax = ref[:, k].min(), ref[:, k].max()
        pad = 1e-4 * max(vmax - vmin, 1e-12)
        vmin, vmax = vmin - pad, vmax + pad
        levels = np.linspace(vmin, vmax, 41)
        for j, (data, title) in enumerate([(ref[:, k], "CFD (OpenFOAM)"), (pred[:, k], "PINN")]):
            ax = axes[k, j]
            # extend="both": values just outside the colour range get the end colours instead of a blank
            c = ax.tricontourf(tri, data, levels=levels, cmap="jet", extend="both")
            ax.set_title(f"{title}: {LABELS[n]}")
            ax.set_aspect("equal")
            fig.colorbar(c, ax=ax, shrink=0.8)
        ax = axes[k, 2]
        err = np.abs(pred[:, k] - ref[:, k])
        c = ax.tricontourf(tri, err, levels=41, cmap="magma")
        ax.set_title(f"|error| {LABELS[n]}")
        ax.set_aspect("equal")
        fig.colorbar(c, ax=ax, shrink=0.8)
    fig.suptitle(f"wedge {math.degrees(domain.theta):.0f} deg {tag}")
    fig.savefig(os.path.join(out_dir, "fields.png"), dpi=130)
    plt.close(fig)


def plot_cuts(model, X_ref, Q_ref, domain, out_dir, xs=None, device="cpu"):
    """Profiles along vertical lines: CFD vs PINN."""
    os.makedirs(out_dir, exist_ok=True)
    xs = xs or domain.stations((0.5, 0.9))
    x = X_ref[:, 0].cpu().numpy(); y = X_ref[:, 1].cpu().numpy(); ref = Q_ref.cpu().numpy()
    y_top = domain.y0 + min(0.7 * domain.L, domain.y1 - domain.y0)
    fig, axes = plt.subplots(len(xs), 4, figsize=(16, 3.6 * len(xs)), constrained_layout=True)
    for i, xc in enumerate(xs):
        # nearest CFD column (blockMesh columns are vertical lines)
        xn = x[np.argmin(np.abs(x - xc))]
        col = np.abs(x - xn) < 1e-6 * domain.L
        o = np.argsort(y[col]); yy = y[col][o]; rr = ref[col][o]
        sel = yy < y_top
        y_w = domain.y0 + (xn - domain.w_start) * math.tan(domain.theta)
        yl = torch.linspace(y_w, y_top, 1500, device=device)
        P = predict(model, torch.stack([torch.full_like(yl, float(xn)), yl], 1)).cpu().numpy()
        for k, n in enumerate(NAMES):
            ax = axes[i, k]
            ax.plot(yy[sel], rr[sel, k], "k-", lw=2, label="CFD")
            ax.plot(yl.cpu().numpy(), P[:, k], "r--", lw=1.5, label="PINN")
            ax.set_title(f"x = {xn:.3f}: {LABELS[n]}")
            ax.set_xlabel("y")
            ax.grid(alpha=0.3)
            if k == 0:
                ax.legend()
    fig.savefig(os.path.join(out_dir, "cuts.png"), dpi=130)
    plt.close(fig)


def plot_history(hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    it = [h["it"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for key in ["loss", "pde", "bc"]:
        axes[0].semilogy(it, [h[key] for h in hist], label=key)
    axes[0].set_xlabel("iteration"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title("training loss")
    if "pertL2_roi" in hist[-1]:
        ev = [h for h in hist if "pertL2_roi" in h]
        for k, n in enumerate(NAMES):
            axes[1].semilogy([h["it"] for h in ev], [h["pertL2_roi"][k] for h in ev], label=n)
        axes[1].set_title("disturbance-relative L2 error near the wedge"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("iteration")
    fig.savefig(os.path.join(out_dir, "history.png"), dpi=130)
    plt.close(fig)
