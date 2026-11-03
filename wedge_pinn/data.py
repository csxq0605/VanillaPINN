"""Geometry, collocation sampling and the OpenFOAM reference solution."""
import math
import os
import numpy as np
import torch

RHO_REF = 1.16838727127
P_REF = 101325.0
VEL_REF = math.sqrt(P_REF / RHO_REF)


class WedgeDomain:
    """Rectangle [x0,x1]x[y0,y1] minus the wedge that starts at (w_start, y0) with angle theta."""

    def __init__(self, x0=0.0, x1=2.0, y0=0.0, y1=2.0, w_start=1.0, theta_deg=10.0):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.w_start = w_start
        self.theta = math.radians(theta_deg)

    @property
    def bounds(self):
        return ((self.x0, self.x1), (self.y0, self.y1))

    def wall_y(self, x):
        """Lower boundary height (symmetry line, then wedge surface)."""
        return self.y0 + torch.clamp(x - self.w_start, min=0.0) * math.tan(self.theta)

    def inside(self, x, y):
        return y >= self.wall_y(x)

    # ---------------- evaluation geometry (scaled with the wedge length) ----------------
    @property
    def L(self):
        return self.x1 - self.w_start

    def stations(self, fracs=(0.25, 0.5, 0.75, 0.95)):
        return [self.w_start + f * self.L for f in fracs]

    def roi_mask(self, x, y):
        """Region around the wedge and the shock, used for the error metrics."""
        return (x >= self.w_start - 0.1 * self.L) & (y <= self.y0 + min(0.6 * self.L, self.y1 - self.y0))

    def plot_window(self):
        return (self.w_start - 0.2 * self.L, self.x1, self.y0, self.y0 + min(0.6 * self.L, self.y1 - self.y0))

    # ---------------- collocation points ----------------
    def sample_uniform(self, n, device):
        """Uniform random points in the fluid region (rejection from the bounding box)."""
        out = []
        need = n
        while need > 0:
            x = self.x0 + (self.x1 - self.x0) * torch.rand(2 * need, device=device)
            y = self.y0 + (self.y1 - self.y0) * torch.rand(2 * need, device=device)
            m = self.inside(x, y)
            out.append(torch.stack([x[m], y[m]], 1))
            need -= int(m.sum())
        return torch.cat(out)[:n]

    def sample_box(self, n, box, device):
        """Uniform points in a sub-box (xa, xb, height above wall)."""
        if n <= 0:
            return torch.empty(0, 2, device=device)
        xa, xb, h = box
        out = []
        need = n
        while need > 0:
            x = xa + (xb - xa) * torch.rand(2 * need, device=device)
            y = self.wall_y(x) + h * torch.rand(2 * need, device=device)
            m = (y <= self.y1)
            out.append(torch.stack([x[m], y[m]], 1))
            need -= int(m.sum())
        return torch.cat(out)[:n]

    # ---------------- finite-volume boxes ----------------
    def sample_boxes(self, n, h_min, h_max, focus, focus_frac, device):
        """Axis-aligned squares fully inside the fluid. Returns centres (n, 2) and sizes (n,).
        Sizes are log-uniform in [h_min, h_max]; `focus_frac` of the centres come from the
        band above the wedge given by focus = (xa, xb, height)."""
        out_c, out_h = [], []
        need = n
        while need > 0:
            m = 3 * need
            h = h_min * (h_max / h_min) ** torch.rand(m, device=device)
            nf = int(focus_frac * m)
            c_f = self.sample_box(nf, focus, device)
            c_u = self.sample_uniform(m - nf, device)
            c = torch.cat([c_f, c_u])
            xl, xr = c[:, 0] - h / 2, c[:, 0] + h / 2
            yb, yt = c[:, 1] - h / 2, c[:, 1] + h / 2
            # wall height is non-decreasing in x, so the bottom-right corner is the critical one
            ok = (xl >= self.x0) & (xr <= self.x1) & (yt <= self.y1) & (yb >= self.wall_y(xr))
            out_c.append(c[ok]); out_h.append(h[ok])
            need -= int(ok.sum())
        return torch.cat(out_c)[:n], torch.cat(out_h)[:n]

    # ---------------- boundary points ----------------
    def boundary(self, n, device):
        """Returns dict of boundary point sets (N, 2)."""
        def lin(a, b, k):
            return a + (b - a) * torch.rand(k, device=device)

        nb = n
        x_in = torch.full((nb,), self.x0, device=device)
        y_in = lin(self.y0, self.y1, nb)
        x_top = lin(self.x0, self.x1, nb)
        y_top = torch.full((nb,), self.y1, device=device)
        x_sym = lin(self.x0, self.w_start, nb)
        y_sym = torch.full((nb,), self.y0, device=device)
        # more wedge points: the wedge is where the flow is turned
        x_w = lin(self.w_start, self.x1, 2 * nb)
        y_w = self.wall_y(x_w)
        return {
            "inlet": torch.stack([x_in, y_in], 1),
            "top": torch.stack([x_top, y_top], 1),
            "symmetry": torch.stack([x_sym, y_sym], 1),
            "wedge": torch.stack([x_w, y_w], 1),
        }


def _read_openfoam_field(path, ncomp):
    with open(path, "r") as f:
        lines = f.readlines()
    for i, s in enumerate(lines):
        if s.strip().startswith("internalField"):
            n = int(lines[i + 1].strip())
            start = i + 3
            break
    rows = []
    for s in lines[start:start + n]:
        s = s.strip().strip("()")
        rows.append([float(t) for t in s.split()][:ncomp])
    return np.asarray(rows, dtype=np.float64)


def blockmesh_cell_centres(case_dir):
    """Cell centres of a 2D blockMesh (one cell in z, simpleGrading (1 1 1)), in OpenFOAM
    cell order: block by block, i fastest. Used when the case has no written C field."""
    import re
    with open(os.path.join(case_dir, "system", "blockMeshDict")) as f:
        txt = re.sub(r"//.*", "", f.read())
    vtxt = re.search(r"vertices\s*\((.*?)\n\s*\);", txt, re.S).group(1)
    verts = np.array([[float(t) for t in m.split()] for m in re.findall(r"\(([^()]*)\)", vtxt)])
    btxt = re.search(r"blocks\s*\((.*?)\n\s*\);", txt, re.S).group(1)
    cells = []
    for hexv, ncell, grading in re.findall(r"hex\s*\(([^)]*)\)\s*\(([^)]*)\)\s*simpleGrading\s*\(([^)]*)\)", btxt):
        v = [int(t) for t in hexv.split()]
        nx, ny, _ = [int(t) for t in ncell.split()]
        if any(abs(float(g) - 1) > 1e-12 for g in grading.split()):
            raise NotImplementedError("only uniform grading is supported")
        P0, P1, P2, P3 = (verts[k, :2] for k in v[:4])
        s = (np.arange(nx) + 0.5) / nx
        t = (np.arange(ny) + 0.5) / ny
        S, T = np.meshgrid(s, t)            # j outer, i inner -> ravel gives i fastest
        S, T = S.ravel()[:, None], T.ravel()[:, None]
        cells.append((1 - S) * (1 - T) * P0 + S * (1 - T) * P1 + S * T * P2 + (1 - S) * T * P3)
    return np.vstack(cells)


def load_reference(case_dir, time="0.005", cache=None, device="cpu", rho_ref=RHO_REF, p_ref=P_REF):
    """OpenFOAM cell-centre solution, non-dimensionalised like the PINN
    (rho / rho_ref, p / p_ref, U / sqrt(p_ref / rho_ref)).

    Returns X (N, 2) and Q (N, 4) = (u, v, rho, p).
    """
    if cache and os.path.exists(cache):
        d = np.load(cache)
        X, Q = d["X"], d["Q"]
    else:
        vel_ref = math.sqrt(p_ref / rho_ref)
        c_file = os.path.join(case_dir, "0", "C")
        C = _read_openfoam_field(c_file, 2) if os.path.exists(c_file) else blockmesh_cell_centres(case_dir)
        U = _read_openfoam_field(os.path.join(case_dir, time, "U"), 2) / vel_ref
        rho = _read_openfoam_field(os.path.join(case_dir, time, "rho"), 1) / rho_ref
        p = _read_openfoam_field(os.path.join(case_dir, time, "p"), 1) / p_ref
        X = C
        Q = np.hstack([U, rho, p])
        if cache:
            np.savez_compressed(cache, X=X, Q=Q)
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Q, dtype=torch.float32, device=device))
