"""Widen a trained parametric network without changing the function it computes (Net2Net-style).

New hidden units get random incoming weights and zero outgoing weights, so the widened network starts from
exactly the old solution and the extra units only grow if the loss wants them. Used to test whether a run
that has stopped improving is limited by network size.

    E:/Conda/envs/cv/python.exe widen_param.py runs_param/<run> 256 runs_param/<run>_w256
    E:/Conda/envs/cv/python.exe param_pinn.py --init_from runs_param/<run>_w256 --width 256 ...
"""
import json
import os
import sys
from dataclasses import asdict

import torch

from param_pinn import ParamPINN, PConfig


@torch.no_grad()
def widen(old: ParamPINN, cfg_new: PConfig) -> ParamPINN:
    new = ParamPINN(cfg_new)
    W, W2 = old.head.in_features, new.head.in_features
    assert W2 >= W and old.arch == new.arch == "hyper"
    ng = old.n_geo
    new.lo.copy_(old.lo); new.hi.copy_(old.hi)
    # parameter-modulated first layer: rows of the base weights and of the hypernetwork output
    new.first.weight[:W] = old.first.weight
    new.first.bias[:W] = old.first.bias
    for a, b in zip(old.hyper[:-1], new.hyper[:-1]):
        if isinstance(a, torch.nn.Linear):
            b.weight.copy_(a.weight); b.bias.copy_(a.bias)
    ho, hn = old.hyper[-1], new.hyper[-1]
    hn.weight.zero_(); hn.bias.zero_()
    hn.weight.view(W2, ng + 1, -1)[:W] = ho.weight.view(W, ng + 1, -1)
    hn.bias.view(W2, ng + 1)[:W] = ho.bias.view(W, ng + 1)
    # body: old units keep their weights and get zero weights from new units
    lo_, ln_ = [m for m in old.body if isinstance(m, torch.nn.Linear)], [m for m in new.body if isinstance(m, torch.nn.Linear)]
    for k, (a, b) in enumerate(zip(lo_, ln_)):
        b.weight[:W].zero_()
        if k == 0:   # input = [first-layer units (W), mu (2)]
            b.weight[:W, :W] = a.weight[:, :W]
            b.weight[:W, W2:] = a.weight[:, W:]
        else:
            b.weight[:W, :W] = a.weight
        b.bias[:W] = a.bias
    new.head.weight.zero_()
    new.head.weight[:, :W] = old.head.weight
    new.head.bias.copy_(old.head.bias)
    return new


def main():
    src, width, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cfg = PConfig(**json.load(open(os.path.join(src, "config.json"))))
    old = ParamPINN(cfg)
    old.load_state_dict(torch.load(os.path.join(src, "model_final.pt"), map_location="cpu", weights_only=True))
    cfg_new = PConfig(**{**asdict(cfg), "width": width})
    new = widen(old, cfg_new)
    # check: same outputs and derivatives
    torch.manual_seed(1)
    n = 2000
    X = torch.stack([torch.rand(n) * 1.25 - 0.25, torch.rand(n) * 0.9, 3 + 3 * torch.rand(n),
                     torch.deg2rad(5 + 15 * torch.rand(n))], 1)
    q0, d0, dd0 = old.forward_derivs(X)
    q1, d1, dd1 = new.forward_derivs(X)
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()
    err = max(rel(q1, q0), rel(d1, d0), rel(dd1, dd0))
    print(f"width {cfg.width} -> {width}, max relative difference (q, dq, d2q) {rel(q1, q0):.1e} {rel(d1, d0):.1e} "
          f"{rel(dd1, dd0):.1e}, params {sum(p.numel() for p in new.parameters())}")
    assert err < 1e-5
    os.makedirs(dst, exist_ok=True)
    torch.save(new.state_dict(), os.path.join(dst, "model_final.pt"))
    json.dump(asdict(cfg_new), open(os.path.join(dst, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
