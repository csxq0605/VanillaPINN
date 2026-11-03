"""Function-preserving widening of a ParamPINN (hyper arch): widen_model.py <src_run> <new_width> <out_dir> [ckpt]
Old units keep their weights; new hidden units get small random incoming weights and ZERO outgoing weights (the
columns that read them in the next layer, and in the head), so the widened network computes exactly the same function
and derivatives. Training then grows the new units from the current solution instead of restarting from scratch."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_pinn import PConfig, ParamPINN

src, W2, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ck = sys.argv[4] if len(sys.argv) > 4 else "model_final.pt"
d = json.load(open(os.path.join(src, "config.json")))
assert d["arch"] == "hyper" and d["coords"] == "cart"
c1 = PConfig(**d); m1 = ParamPINN(c1)
m1.load_state_dict(torch.load(os.path.join(src, ck), map_location="cpu", weights_only=True)); m1.eval()
d2 = dict(d, width=W2); c2 = PConfig(**d2); m2 = ParamPINN(c2)
W1, ng = c1.width, m1.n_geo
torch.manual_seed(0)
eps = 0.05
with torch.no_grad():
    m2.lo.copy_(m1.lo); m2.hi.copy_(m1.hi)
    # first layer: old rows copied, new rows small random
    m2.first.weight.normal_(0, eps); m2.first.bias.normal_(0, eps)
    m2.first.weight[:W1] = m1.first.weight; m2.first.bias[:W1] = m1.first.bias
    # hyper net: hidden layers unchanged; output (width*(ng+1)) laid out as (width, ng+1): copy old units, new = 0
    for i in (0, 2):
        m2.hyper[i].weight.copy_(m1.hyper[i].weight); m2.hyper[i].bias.copy_(m1.hyper[i].bias)
    m2.hyper[4].weight.zero_(); m2.hyper[4].bias.zero_()
    m2.hyper[4].weight.view(W2, ng + 1, -1)[:W1] = m1.hyper[4].weight.view(W1, ng + 1, -1)
    m2.hyper[4].bias.view(W2, ng + 1)[:W1] = m1.hyper[4].bias.view(W1, ng + 1)
    # body: first body layer input = [t (width), mu (2)]
    L1, L2 = list(m1.body)[0::2], list(m2.body)[0::2]
    for k, (a, b) in enumerate(zip(L1, L2)):
        b.weight.normal_(0, eps / (b.weight.shape[1] ** 0.5) * 4); b.bias.normal_(0, eps)
        b.bias[:W1] = a.bias
        b.weight[:W1].zero_()                        # old units: read nothing from new units
        b.weight[:W1, :W1] = a.weight[:, :W1]
        if k == 0:                                   # mu columns
            b.weight[:W1, W2:W2 + 2] = a.weight[:, W1:W1 + 2]
    m2.head.weight.zero_(); m2.head.weight[:, :W1] = m1.head.weight; m2.head.bias.copy_(m1.head.bias)
# check: same values and derivatives
X = torch.stack([torch.rand(3000) * 1.25 - 0.25, torch.rand(3000) * c1.y1,
                 c1.ma_min + (c1.ma_max - c1.ma_min) * torch.rand(3000), torch.deg2rad(c1.th_min + (c1.th_max - c1.th_min) * torch.rand(3000))], 1)
q1, dq1, d2q1 = m1.forward_derivs(X); q2, dq2, d2q2 = m2.forward_derivs(X)
rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()
print(f"widen {W1} -> {W2}: max rel diff q {rel(q2, q1):.2e}, dq {rel(dq2, dq1):.2e}, d2q {rel(d2q2, d2q1):.2e}")
os.makedirs(out, exist_ok=True)
torch.save(m2.state_dict(), os.path.join(out, "model_final.pt"))
json.dump(d2, open(os.path.join(out, "config.json"), "w"), indent=2)
print("saved", out)
