"""Physics for 2D steady inviscid supersonic flow over a wedge.

Non-dimensionalisation (same as the OpenFOAM post-processing in vanilla_Pinn):
    rho / rho_inf, p / p_inf, velocity / sqrt(p_inf / rho_inf)
so the free stream is rho = 1, p = 1, u = sqrt(gamma) * M, v = 0.
"""
import math
import torch
from scipy.optimize import brentq

GAMMA = 1.4


class Freestream:
    def __init__(self, mach=5.0, gamma=GAMMA):
        self.gamma = gamma
        self.mach = mach
        self.rho = 1.0
        self.p = 1.0
        self.u = math.sqrt(gamma * self.p / self.rho) * mach
        self.v = 0.0

    def state(self):
        """(u, v, rho, p)"""
        return [self.u, self.v, self.rho, self.p]


def oblique_shock(mach, theta, gamma=GAMMA):
    """Weak oblique-shock solution. Returns beta (rad) and post-shock (u, v, rho, p)."""
    def f(b):
        return math.tan(theta) - 2.0 / math.tan(b) * (mach**2 * math.sin(b)**2 - 1) / (
            mach**2 * (gamma + math.cos(2 * b)) + 2)

    beta = brentq(f, math.asin(1.0 / mach) + 1e-8, math.radians(64.0))
    mn = mach * math.sin(beta)
    p2 = 1 + 2 * gamma / (gamma + 1) * (mn**2 - 1)
    rho2 = (gamma + 1) * mn**2 / ((gamma - 1) * mn**2 + 2)
    mn2 = math.sqrt((1 + (gamma - 1) / 2 * mn**2) / (gamma * mn**2 - (gamma - 1) / 2))
    m2 = mn2 / math.sin(beta - theta)
    speed2 = m2 * math.sqrt(gamma * p2 / rho2)
    return beta, [speed2 * math.cos(theta), speed2 * math.sin(theta), rho2, p2]


def exact_solution(x, y, fs, w_start, theta, y0=0.0):
    """Piecewise-constant exact solution, valid while the shock leaves through the outlet."""
    beta, post = oblique_shock(fs.mach, theta, fs.gamma)
    inside = (x > w_start) & (y < y0 + (x - w_start) * math.tan(beta))
    out = torch.empty(x.shape[0], 4, dtype=x.dtype, device=x.device)
    for k, (a, b) in enumerate(zip(fs.state(), post)):
        out[:, k] = torch.where(inside.view(-1), torch.full_like(out[:, k], b), torch.full_like(out[:, k], a))
    return out


def euler_residual(q, X, gamma=GAMMA, nu=0.0):
    """Conservative residual dF/dx + dG/dy of the 2D Euler equations.

    q: (N, 4) primitive state (u, v, rho, p) produced from X (N, 2) with requires_grad.
    Flux divergences are assembled from primitive derivatives by the chain rule, which is
    exactly the divergence of the conservative flux (so the conservative property is kept).
    nu > 0 adds an artificial-viscosity term -nu * Laplace(U) on the conservative variables.
    Returns residual (N, 4) and velocity divergence (N,).
    """
    grads = [torch.autograd.grad(q[:, k].sum(), X, create_graph=True)[0] for k in range(4)]
    u, v, r, p = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    (ux, uy), (vx, vy), (rx, ry), (px, py) = [(g[:, 0], g[:, 1]) for g in grads]
    g1 = gamma / (gamma - 1.0)
    k2 = u * u + v * v
    k2x = 2 * (u * ux + v * vx)
    k2y = 2 * (u * uy + v * vy)

    # x-fluxes F = (ru, ru^2+p, ruv, g1*p*u + 0.5*r*u*k2)
    ru_x = rx * u + r * ux
    F1x = ru_x
    F2x = ru_x * u + r * u * ux + px
    F3x = ru_x * v + r * u * vx
    F4x = g1 * (px * u + p * ux) + 0.5 * (ru_x * k2 + r * u * k2x)
    # y-fluxes G = (rv, ruv, rv^2+p, g1*p*v + 0.5*r*v*k2)
    rv_y = ry * v + r * vy
    G1y = rv_y
    G2y = rv_y * u + r * v * uy
    G3y = rv_y * v + r * v * vy + py
    G4y = g1 * (py * v + p * vy) + 0.5 * (rv_y * k2 + r * v * k2y)

    res = torch.stack([F1x + G1y, F2x + G2y, F3x + G3y, F4x + G4y], dim=1)

    if nu > 0.0:
        # conservative variables and their Laplacians
        U = torch.stack([r, r * u, r * v, p / (gamma - 1.0) + 0.5 * r * k2], dim=1)
        lap = []
        for k in range(4):
            gU = torch.autograd.grad(U[:, k].sum(), X, create_graph=True)[0]
            Uxx = torch.autograd.grad(gU[:, 0].sum(), X, create_graph=True)[0][:, 0]
            Uyy = torch.autograd.grad(gU[:, 1].sum(), X, create_graph=True)[0][:, 1]
            lap.append(Uxx + Uyy)
        res = res - nu * torch.stack(lap, dim=1)

    return res, ux + vy


def viscous_euler_residual(q, dq, d2q, gamma=GAMMA, nu=0.0):
    """div(F, G) - nu * Laplace(U) from precomputed derivatives (see WedgePINN.forward_derivs).

    q (N,4) primitive (u, v, rho, p); dq, d2q (2,N,4): first / pure second derivatives in x, y.
    nu: scalar or (N,) artificial viscosity acting on the conservative variables U.
    Returns residual (N,4) and velocity divergence (N,).
    """
    u, v, r, p = q.unbind(1)
    (ux, vx, rx, px), (uy, vy, ry, py) = dq[0].unbind(1), dq[1].unbind(1)
    g1 = gamma / (gamma - 1.0)
    k2 = u * u + v * v
    k2x, k2y = 2 * (u * ux + v * vx), 2 * (u * uy + v * vy)
    ru_x = rx * u + r * ux
    rv_y = ry * v + r * vy
    res = torch.stack([
        ru_x + rv_y,
        ru_x * u + r * u * ux + px + rv_y * u + r * v * uy,
        ru_x * v + r * u * vx + rv_y * v + r * v * vy + py,
        g1 * (px * u + p * ux) + 0.5 * (ru_x * k2 + r * u * k2x)
        + g1 * (py * v + p * vy) + 0.5 * (rv_y * k2 + r * v * k2y),
    ], 1)
    if d2q is not None and (torch.is_tensor(nu) or nu > 0):
        lap = 0.0
        for d in range(2):
            (u1, v1, r1, p1), (u2, v2, r2, p2) = dq[d].unbind(1), d2q[d].unbind(1)
            k1 = 2 * (u * u1 + v * v1)
            kk = 2 * (u1 * u1 + u * u2 + v1 * v1 + v * v2)
            lap = lap + torch.stack([
                r2,
                r2 * u + 2 * r1 * u1 + r * u2,
                r2 * v + 2 * r1 * v1 + r * v2,
                p2 / (gamma - 1.0) + 0.5 * (r2 * k2 + 2 * r1 * k1 + r * kk),
            ], 1)
        nu_t = nu.unsqueeze(1) if torch.is_tensor(nu) else nu
        res = res - nu_t * lap
    return res, ux + vy


def conservative_gradient(q, dq, gamma=GAMMA):
    """dU/dx, dU/dy (2, N, 4) of U = (rho, rho u, rho v, E) from primitive q and dq (2, N, 4)."""
    u, v, r, p = q.unbind(1)
    k2 = u * u + v * v
    out = []
    for d in range(2):
        u1, v1, r1, p1 = dq[d].unbind(1)
        k1 = 2 * (u * u1 + v * v1)
        out.append(torch.stack([r1, r1 * u + r * u1, r1 * v + r * v1,
                                p1 / (gamma - 1.0) + 0.5 * (r1 * k2 + r * k1)], 1))
    return torch.stack(out)


def fluxes(q, gamma=GAMMA):
    """Conservative x- and y-fluxes F, G (N, 4) from primitive q = (u, v, rho, p)."""
    u, v, r, p = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    h = gamma / (gamma - 1.0) * p + 0.5 * r * (u * u + v * v)   # rho * total enthalpy
    F = torch.stack([r * u, r * u * u + p, r * u * v, h * u], 1)
    G = torch.stack([r * v, r * u * v, r * v * v + p, h * v], 1)
    return F, G


def flux_scales(fs):
    """Characteristic magnitude of each conservative flux in the free stream, used to
    balance the four equations (energy flux is ~20x the mass flux otherwise)."""
    g1 = fs.gamma / (fs.gamma - 1.0)
    mass = fs.rho * fs.u
    mom = fs.rho * fs.u**2 + fs.p
    energy = g1 * fs.p * fs.u + 0.5 * fs.rho * fs.u**3
    return [mass, mom, mom, energy]
