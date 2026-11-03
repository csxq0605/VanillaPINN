"""Shock-angle error over a finer (Ma, theta) grid, to see whether errors sit on the edges of the training box.
edge_scan.py <run_dir> [ckpt]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from param_pinn import PConfig, ParamPINN, evaluate_case

run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else ("model_final.pt" if os.path.exists(os.path.join(run, "model_final.pt")) else "model_last.pt")
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
mas = [3.0, 3.25, 3.5, 4.0, 5.0, 5.5, 6.0]
ths = [5.0, 7.5, 12.5, 17.5, 20.0]
print("beta error (deg), rows Ma, cols theta " + " ".join(f"{t:6.1f}" for t in ths))
for ma in mas:
    row = []
    for t in ths:
        ev = evaluate_case(m, cfg, ma, t, torch.device("cpu"), nx=120, ny=90)
        row.append(ev["beta"] - ev["beta_exact"])
    print(f"Ma {ma:4.2f}                              " + " ".join(f"{v:+6.2f}" for v in row))
