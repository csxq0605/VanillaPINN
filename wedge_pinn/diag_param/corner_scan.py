"""Shock-angle error on a fine parameter grid near the low-Ma / high-theta corner: corner_scan.py <run_dir> [ckpt]"""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from param_pinn import PConfig, ParamPINN, evaluate_case
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else "model_final.pt"
cfg = PConfig(**json.load(open(os.path.join(run, "config.json"))))
m = ParamPINN(cfg); m.load_state_dict(torch.load(os.path.join(run, ck), map_location="cpu", weights_only=True)); m.eval()
cpu = torch.device("cpu")
mas = [2.75, 2.85, 3.0, 3.1, 3.25, 3.5]; ths = [15.0, 17.5, 19.0, 20.0, 20.5]
print(f"training box: Ma {cfg.ma_min}-{cfg.ma_max}, theta {cfg.th_min}-{cfg.th_max}, y1 {cfg.y1}")
print("beta err".ljust(10) + "".join(f"{t:8.1f}" for t in ths))
for ma in mas:
    row = [evaluate_case(m, cfg, ma, t, cpu, nx=40, ny=30) for t in ths]
    print(f"Ma {ma:5.2f} " + "".join(f"{e['beta'] - e['beta_exact']:+8.2f}" for e in row), flush=True)
