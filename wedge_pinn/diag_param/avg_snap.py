"""Iterate averaging: average the weight snapshots model_s*.pt of a run into model_avg.pt.
avg_snap.py <run_dir> [first_step]   (only snapshots with step >= first_step; default all)
Resampling and L-BFGS restarts make the iterates jitter around a minimum; averaging them (Polyak / SWA) reduces that
variance without using any information other than the training trajectory."""
import sys, os, glob, re, torch
run = sys.argv[1]; first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
files = sorted(f for f in glob.glob(os.path.join(run, "model_s*.pt")) if int(re.findall(r"model_s(\d+)", f)[0]) >= first)
assert files, "no snapshots"
sds = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
avg = {}
for k, v in sds[0].items():
    if v.is_floating_point() and k not in ("lo", "hi"):
        avg[k] = sum(sd[k] for sd in sds) / len(sds)
    else:
        avg[k] = v
torch.save(avg, os.path.join(run, "model_avg.pt"))
print(f"averaged {len(files)} snapshots: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
