#!/bin/bash
# Re-run the training chain that produced the released parametric model (models/param_B1_w192).
# Each stage warm-starts from the previous one (--init_from). Stage lengths follow docs/ablations.md; the original
# runs were sometimes cut short by GPU faults, so a re-run will land close to, not exactly on, the published numbers.
# About 12-14 h on one RTX 4060 (8 GB). Run ONE stage at a time on an 8 GB card.
#   cd wedge_pinn && bash scripts/train_param.sh [first_stage]
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
last() { ls -d runs_param/*_"$1" | tail -1; }
W0="--arch hyper --nu_mode const --adam_iters 0 --line_search strong_wolfe --lbfgs_lr 1 --lr_gamma 1"
W="$W0 --alm true --eq_scale own --w_bc 25 --tip_delta 0.01 --steps 500"
BOX27="--ma_min 2.7 --ma_max 6.25 --th_min 3.5 --th_max 21"
BOX25="--y1 1.05 --ma_min 2.5 --ma_max 6.25 --th_min 3.5 --th_max 21"
START=${1:-E}; run=0
stage() {  # stage <tag> <args...>
  local tag=$1; shift
  [ "$tag" = "$START" ] && run=1
  [ $run = 1 ] || return 0
  echo "=== $tag: $*"; $PY param_pinn.py "$@" --tag "$tag"
}
stage E   --arch hyper --nu_mode const --nu 1e-2 --steps 250                                   # hyper first layer, Adam + L-BFGS
stage E2  --init_from $(last E 2>/dev/null)  $W0 --nu 1e-2 --steps 750                          # strong-Wolfe line search
stage G2  --init_from $(last E2 2>/dev/null) $W0 --nu 1e-2 --w_bc 10 --steps 500                # boundary weight 1 -> 10
stage H2  --init_from $(last G2 2>/dev/null) $W0 --nu 6e-3 --w_bc 16 --steps 500
stage I2  --init_from $(last H2 2>/dev/null) $W0 --nu 4e-3 --w_bc 25 --steps 250
stage J1  --init_from $(last I2 2>/dev/null) $W0 --alm true --nu 4e-3 --w_bc 25 --steps 500    # wall augmented Lagrangian
stage M3  --init_from $(last J1 2>/dev/null) $W0 --alm true --eq_scale own --nu 4e-3 --w_bc 25 --tip_delta 0.01 --steps 750  # per-law flux scaling
stage O2  --init_from $(last M3 2>/dev/null) $W --nu 3e-3 --steps 250
stage P1  --init_from $(last O2 2>/dev/null) $W --nu 3e-3 --ma_min 2.85 --ma_max 6.25 --th_min 3.5 --th_max 21 --steps 200   # wider training box
stage Q1  --init_from $(last P1 2>/dev/null) $W --nu 2.5e-3 --ma_min 2.85 --ma_max 6.25 --th_min 3.5 --th_max 21
stage R1  --init_from $(last Q1 2>/dev/null) $W --nu 2e-3   --ma_min 2.85 --ma_max 6.25 --th_min 3.5 --th_max 21
stage S1  --init_from $(last R1 2>/dev/null) $W --nu 1.5e-3 --ma_min 2.85 --ma_max 6.25 --th_min 3.5 --th_max 21
stage T1  --init_from $(last S1 2>/dev/null) $W --nu 1.5e-3 $BOX27
stage U1  --init_from $(last T1 2>/dev/null) $W --nu 1e-3   $BOX27
stage V1  --init_from $(last U1 2>/dev/null) $W --nu 7e-4   $BOX27
stage W1  --init_from $(last V1 2>/dev/null) $W --nu 7e-4 --n_rad 22000 $BOX27                  # more RAD points
stage A1  --init_from $(last W1 2>/dev/null) $W --nu 7e-4 --n_rad 22000 $BOX25
stage A2  --init_from $(last A1 2>/dev/null) $W --nu 7e-4 --n_pts 31000 --n_rad 27500 --rad_pool 140000 $BOX25 --steps 400
if [ $run = 1 ] || [ "$START" = "B1" ]; then
  $PY widen_model.py $(last A2) 192 runs_param/widen_A2_w192
  START=B1; run=0
fi
stage B1  --init_from runs_param/widen_A2_w192 $W --width 192 --nu 7e-4 --n_pts 31000 --n_rad 27500 --rad_pool 140000 $BOX25
echo "final model: $(last B1)/model_final.pt  (evaluate in y <= 0.9: copy config.json with y1 = 0.9, see models/param_B1_w192)"
