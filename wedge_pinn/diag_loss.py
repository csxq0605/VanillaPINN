"""Where are the PDE residual and the box-conservation residual of a capture run concentrated? (diagnostic)"""
import sys, json, os, math, torch
from dataclasses import fields
import capture_lv as C

run = sys.argv[1]
cfg = json.load(open(os.path.join(run, "config.json")))
keep = {f.name for f in fields(C.LVConfig)}
cfg = {k: v for k, v in cfg.items() if k in keep}
cfg.update(init_from=run, out_root=sys.argv[2], tag="diag", n_box=6000)
cap = C.LVCapture(C.LVConfig(**cfg))
d, dev = cap.dom, cap.device
beta = cap.beta
torch.manual_seed(0)
X = d.sample_uniform(60000, dev)
with torch.no_grad():
    r = torch.cat([cap.residual(X[i:i+20000])[0].pow(2).sum(1) for i in range(0, len(X), 20000)])
cap.new_boxes()
ctr, h, s = cap.boxes
with torch.no_grad():
    nq = cap.cfg.n_quad
    hx = h[:, None]; xc, yc = ctr[:, 0:1], ctr[:, 1:2]; z = 0 * s
    pts = torch.cat([torch.stack([xc + hx/2 + z, yc + hx*s], -1), torch.stack([xc - hx/2 + z, yc + hx*s], -1),
                     torch.stack([xc + hx*s, yc + hx/2 + z], -1), torch.stack([xc + hx*s, yc - hx/2 + z], -1)], 1).reshape(-1, 2)
    F, G = C.fluxes(cap.model(pts), cap.fs.gamma)
    n = ctr.shape[0]
    F = F.reshape(n, 4, nq, 4); G = G.reshape(n, 4, nq, 4)
    bal = F[:, 0].mean(1) - F[:, 1].mean(1) + G[:, 2].mean(1) - G[:, 3].mean(1)
    wb = ((bal / torch.tensor(C.flux_scales(cap.fs), device=dev)) ** 2).sum(1)

def regions(P):
    x, y = P[:, 0], P[:, 1]
    ys = d.y0 + (x - d.w_start).clamp(min=0) * math.tan(beta)
    yw = d.wall_y(x)
    tip = ((x - d.w_start) ** 2 + (y - d.y0) ** 2).sqrt() < 0.1 * d.L
    shock = (x > d.w_start) & ((y - ys).abs() < 0.05 * d.L) & ~tip
    wall = (x > d.w_start) & ((y - yw) < 0.05 * d.L) & ~tip & ~shock
    top = (y > d.y1 - 0.05 * d.L) & ~shock
    inlet = x < d.x0 + 0.05 * d.L
    post = (x > d.w_start) & (y < ys) & ~tip & ~shock & ~wall
    rest = ~(tip | shock | wall | top | inlet | post)
    return dict(tip=tip, shock_band=shock, wall=wall, post_shock=post, top=top, inlet=inlet, upstream_rest=rest)

print(run)
print(f"{'region':14s} {'pts%':>6s} {'PDE share':>9s} {'boxes%':>7s} {'weak share':>10s}")
R1, R2 = regions(X), regions(ctr)
for k in R1:
    print(f"{k:14s} {100*R1[k].float().mean():6.1f} {100*r[R1[k]].sum()/r.sum():9.1f} {100*R2[k].float().mean():7.1f} {100*wb[R2[k]].sum()/wb.sum():10.1f}")
print(f"total: mean PDE residual^2 {r.mean():.3e}, mean box imbalance^2 {wb.mean():.3e}")
