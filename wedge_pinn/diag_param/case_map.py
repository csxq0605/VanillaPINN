"""2D error / residual maps for a few cases: case_map.py <run_dir> [ckpt]
Columns: p - p_exact, flow angle - theta, entropy - exact post-shock entropy, |PDE residual| (relative to own
free-stream flux, 1/L, log10). Also prints the pressure and flow angle along lines parallel to the wall at fixed
fractions of the wall-to-shock distance, to see where the post-shock pressure deficit starts."""
import sys, os, json, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import param_pinn as P

run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else "model_final.pt"
d = json.load(open(os.path.join(run, "config.json")))
d.update(init_from="", rad_pool=1000, out_root=os.path.join(tempfile.gettempdir(), "param_diag_runs"))
T = P.ParamTrainer(P.PConfig(**d))
T.model.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); T.model.eval()
m, cfg, cpu = T.model, T.cfg, torch.device("cpu")
G = 1.4
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
cases = [(3.0, 20.0), (4.0, 20.0), (6.0, 20.0), (3.0, 10.0)]
fig, axs = plt.subplots(len(cases), 4, figsize=(18, 3.6 * len(cases)))
for row, (ma, td) in enumerate(cases):
    th = math.radians(td); be, post = P.oblique_shock(ma, th); post = torch.tensor(post)
    s_post = math.log(post[3] / post[2] ** G)
    nx, ny = 200, 150
    xg, yg = torch.meshgrid(torch.linspace(-0.05, cfg.x1, nx), torch.linspace(0, 0.9, ny), indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    q = P.predict(m, x, y, ma, th, cpu)
    wall = y < x.clamp(min=0) * math.tan(th)
    inside = (x > 0) & (y < x * math.tan(be)) & ~wall
    pe = torch.where(inside, post[3], torch.tensor(1.0))
    dp = (q[:, 3] - pe) / post[3]
    ang = torch.rad2deg(torch.atan2(q[:, 1], q[:, 0])) - td
    ent = torch.log(q[:, 3] / q[:, 2] ** G) - s_post
    for a in (dp, ang, ent):
        a[wall] = float("nan")
    ang[~inside] = float("nan"); ent[~inside] = float("nan")
    # residual on a coarser grid
    rx, ry = torch.meshgrid(torch.linspace(-0.05, cfg.x1, 100), torch.linspace(0, 0.9, 75), indexing="ij")
    rx, ry = rx.reshape(-1), ry.reshape(-1)
    X = torch.stack([rx, ry, torch.full_like(rx, ma), torch.full_like(rx, th)], 1)
    R = []
    for i in range(0, len(X), 2500):
        r, _ = T.residual(X[i:i + 2500]); R.append(r.detach())
    R = torch.cat(R) / P.u_inf(cfg.ma_ref)
    rmag = torch.log10(R.norm(dim=1) + 1e-8)
    rmag[ry < rx.clamp(min=0) * math.tan(th)] = float("nan")
    ext = [-0.05, cfg.x1, 0, 0.9]
    for col, (A, shape, title, cmap, lim) in enumerate([
            (dp, (nx, ny), "(p − p精确) / p₂", "RdBu_r", 0.03),
            (ang, (nx, ny), "气流角 − θ (°)", "RdBu_r", 0.5),
            (ent, (nx, ny), "熵 − 精确激波后熵", "RdBu_r", 0.02),
            (rmag, (100, 75), "log10 |方程残差|", "viridis", None)]):
        ax = axs[row, col]
        kw = dict(vmin=-lim, vmax=lim) if lim else dict(vmin=-3.5, vmax=0.5)
        im = ax.imshow(A.reshape(shape).T.numpy(), origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
        ax.plot([0, cfg.x1], [0, cfg.x1 * math.tan(be)], "k--", lw=0.6)
        ax.set_title(f"Ma {ma:g}, θ {td:g}°: {title}", fontsize=9); fig.colorbar(im, ax=ax)
    # lines at fixed fractions of the wall-to-exact-shock distance
    print(f"== Ma {ma:g} th {td:g}  (exact p2 {post[3]:.3f}, beta {math.degrees(be):.2f})")
    for f in (0.1, 0.5, 0.8, 0.9):
        xs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        ys = xs * ((1 - f) * math.tan(th) + f * math.tan(be))
        qq = P.predict(m, xs, ys, ma, th, cpu)
        pr = " ".join(f"{100 * (v / post[3] - 1):+5.2f}" for v in qq[:, 3].tolist())
        an = " ".join(f"{a:+5.2f}" for a in (torch.rad2deg(torch.atan2(qq[:, 1], qq[:, 0])) - td).tolist())
        print(f"  frac {f:.1f}: p err % at x=0.1..0.9: {pr} | flow angle - theta: {an}")
fig.tight_layout(); out = os.path.join(run, "case_maps.png"); fig.savefig(out, dpi=110); print(out)
