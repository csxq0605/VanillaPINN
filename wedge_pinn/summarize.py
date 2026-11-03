"""Collect capture and refine results of the current recipe into comparison/capture_summary.{md,json}.

    python summarize.py

Edit RUNS below when new runs are added. Order of the 4 numbers: u, v, rho, p.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (case, label, capture run, refine run or None)
RUNS = [
    ("10°", "种子 0（最好的一次）", "2026-10-26_23-50-46_lvf_2e-3", "2026-10-27_01-01-01_refine_lvf"),
    ("10°", "种子 1", "2026-10-27_20-59-04_lvf_seed1", "2026-10-27_23-18-04_refine_seed1"),
    ("10°", "种子 2", "2026-10-27_22-29-09_lvf_seed2", "2026-10-27_23-37-10_refine_seed2"),
    ("10°", "均匀随机配点（代替 Fluent 网格点）", "2026-10-28_11-11-04_lvf_random", None),
    ("15°", "同一配方", "2026-10-27_00-02-16_lvf15", "2026-10-27_01-08-56_refine_lvf15"),
    ("15°", "守恒型粘性项", "2026-10-27_20-58-47_lvc15", "2026-10-27_23-24-09_refine_lvc15"),
    ("15°", "不降权 d_coef 0，ν 下限 1e-2", "2026-10-28_10-05-36_lv15_d0_nu1e-2", None),
    ("15°", "不降权 d_coef 0，ν 下限 5e-3", "2026-10-28_10-06-06_lv15_d0_nu5e-3", None),
    ("15°", "降权系数 d_coef 0.06", "2026-10-28_12-14-12_lv15_d06full", None),
    ("15°", "降权系数 d_coef 0.03", "2026-10-28_12-13-43_lv15_d03", None),
    ("15°", "自适应方框 50%（按压缩量放置），w_weak 10", "2026-10-28_13-17-59_lv15_abox_w10", None),
    ("15°", "自适应方框 50%，w_weak 100", "2026-10-28_13-18-30_lv15_abox_w100", None),
    ("15°", "两阶段：从 d_coef 0.06 接着训，ν 下限 1e-3", "2026-10-28_15-53-14_lv15_s2_nu1e-3", None),
    ("15°", "两阶段：从 d_coef 0.06 接着训，ν 下限 5e-4", "2026-10-28_15-53-34_lv15_s2_nu5e-4", None),
    ("15°（方形域，对精确解）", "同一配方，10° 的方形计算域", "2026-10-28_11-11-24_lv15_square", None),
    ("15°（只用方程，对精确解）", "不降权，可学习粘性 ν≥5e-3，2000 步", "2026-10-28_21-33-26_eqd0_15_long", None),
    ("15°（只用方程，对精确解）", "+ 楔尖权重（边界 + 方程）", "2026-10-28_22-58-48_tip03_all", None),
    ("15°（只用方程，对精确解）", "+ 楔尖权重（只加边界）", "2026-10-28_22-59-18_tip03_bc", None),
    ("15°（只用方程，对精确解）", "接着训：整体粘性缩小 + 残差加点", "2026-10-29_01-00-31_follow_1e-3_rad", None),
    ("15°（只用方程，对精确解）", "接着训：常数粘性 8e-3→2e-3", "2026-10-29_02-02-14_const_8e-3_2e-3", None),
    ("15°（只用方程，对精确解）", "接着训：常数粘性 2e-3 再 1200 步", "2026-10-29_03-10-19_const2e-3_more", None),
    ("15°（只用方程，对精确解）", "接着训：常数粘性 4e-3（约 1100 步，崩溃前存档）", "2026-10-29_04-12-44_eval_const4e-3", None),
]


def load(run):
    with open(os.path.join(HERE, "runs", run, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{100 * x:.2f}%"


def sci(v):
    return ", ".join(f"{x:.1e}" for x in v)


def main():
    rows, out = [], []
    for case, label, cap, ref in RUNS:
        m = load(cap)
        r = load(ref) if ref else None
        rec = dict(case=case, label=label, capture=cap, refine=ref,
                   capture_relL2_vs_cfd=m["relL2_full"], capture_beta=m["beta_pinn"], beta_exact=m["beta_exact"],
                   capture_pertL2_vs_exact=m["pertL2_roi_vs_exact"])
        if r:
            rec.update(refine_beta_stations=r["beta_stations"], refine_pertL2_vs_exact=r["pertL2_roi_vs_exact"])
        out.append(rec)
        u, v, rho, p = m["relL2_full"]
        refined = (f"{min(r['beta_stations']):.3f}–{max(r['beta_stations']):.3f}° | {sci(r['pertL2_roi_vs_exact'])}"
                   if r else "— | —")
        rows.append(f"| {case} | {label} | {pct(rho)} | {pct(p)} | {pct(v)} | {m['beta_pinn']:.2f}°（精确 {m['beta_exact']:.3f}°）"
                    f" | {refined} |")
    head = ("| 算例 | 设置 | 捕捉：ρ 相对 L2（对 CFD） | 捕捉：p 相对 L2 | 捕捉：v 相对 L2 | 捕捉：激波角 "
            "| 精修：激波角（4 测站） | 精修：对精确解扰动 L2（u, v, ρ, p） |\n|---|---|---|---|---|---|---|---|")
    md = head + "\n" + "\n".join(rows) + "\n"
    os.makedirs(os.path.join(HERE, "comparison"), exist_ok=True)
    with open(os.path.join(HERE, "comparison", "capture_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(HERE, "comparison", "capture_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(md)


if __name__ == "__main__":
    main()
