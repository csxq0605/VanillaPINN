"""Shock angle measured more carefully: beta_precise.py <run_dir> [ckpt]
- 50% pressure crossing on vertical lines x = 0.5 / 0.75 / 0.95, 4000 samples, linearly interpolated
  (evaluate_case takes the highest grid point above the level: biased up by ~half a cell, ~+0.04 deg);
- chord angle (mean of atan(y/x)) and the slope of a straight-line fit (direction only, no offset);
- the angle implied by the PINN's own post-shock pressure (p2/p1 -> normal Mach -> beta), an independent check;
- sanity check: the same crossing applied to the exact solution smeared by a tanh must give the exact angle."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN, predict, oblique_shock, eval_cases

GAM = 1.4
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else "model_final.pt"
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
cpu = torch.device("cpu")


def crossing(yy, p):
    """Interpolated y of the 50% level between the wall plateau and p_inf = 1 (last crossing from above)."""
    pw = p[: max(2, len(p) // 40)].mean().item()
    lev = 1 + 0.5 * (pw - 1)
    above = (p > lev).nonzero()
    if len(above) == 0 or above[-1].item() + 1 >= len(p):
        return float("nan"), pw
    i = above[-1].item()
    y0, y1, p0, p1 = yy[i].item(), yy[i + 1].item(), p[i].item(), p[i + 1].item()
    return y0 + (p0 - lev) / (p0 - p1) * (y1 - y0), pw


def beta_from_p(ma, p2):
    """Weak-branch shock angle whose pressure ratio is p2 (p1 = 1)."""
    mn2 = (p2 - 1) * (GAM + 1) / (2 * GAM) + 1
    return math.degrees(math.asin(min(1.0, math.sqrt(max(mn2, 0)) / ma)))


xs = (0.5, 0.75, 0.95)
rows = []
print(" Ma  th  | exact  | chord old-style -> interp | line-fit slope | from post p | smeared-exact check")
for ma, td in eval_cases(cfg):
    th = math.radians(td); be, post = oblique_shock(ma, th); bed = math.degrees(be)
    ys, chk = [], []
    for x in xs:
        yy = torch.linspace(x * math.tan(th), cfg.y1, 4000)
        p = predict(m, torch.full_like(yy, x), yy, ma, th, cpu)[:, 3]
        y50, pw = crossing(yy, p); ys.append(y50)
        # tanh-smeared exact profile with thickness 0.01: the crossing must return the exact shock
        d = (yy - x * math.tan(be)) * math.cos(be)
        pe = 1 + (post[3] - 1) * 0.5 * (1 - torch.tanh(d / 0.0046))
        chk.append(math.degrees(math.atan2(crossing(yy, pe)[0], x)))
    chord = sum(math.degrees(math.atan2(y, x)) for x, y in zip(xs, ys)) / 3
    mx, my = sum(xs) / 3, sum(ys) / 3
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    slope = math.degrees(math.atan(k))
    # post-shock pressure halfway between wall and exact shock, x in [0.6, 0.9] (as evaluate_case)
    xp = torch.linspace(0.6, 0.9, 30); yp = xp * 0.5 * (math.tan(th) + math.tan(be))
    p2 = predict(m, xp, yp, ma, th, cpu)[:, 3].mean().item()
    bp = beta_from_p(ma, p2)
    ck_err = max(abs(c - bed) for c in chk)
    rows.append((chord - bed, slope - bed, bp - bed))
    print(f"{ma:3.0f} {td:4.1f} | {bed:6.2f} | {chord - bed:+.3f} | slope {slope - bed:+.3f} | post-p {bp - bed:+.3f} | check {ck_err:.4f}")
for i, name in enumerate(("chord (interp)", "line-fit slope", "from post-shock p")):
    v = [abs(r[i]) for r in rows]
    print(f"{name:18s}: mean |err| {sum(v) / len(v):.3f}  max {max(v):.3f}  mean signed {sum(r[i] for r in rows) / len(rows):+.3f}")
