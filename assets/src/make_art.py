"""Generate the README artwork (hand-laid SVG): assets/hero_{en,cn}.svg and assets/compare_{en,cn}.svg.

    python assets/src/make_art.py
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "wedge_pinn"))
from physics import oblique_shock  # noqa: E402

BG, PANEL, LINE = "#0b1220", "#121b2f", "#24314d"
TXT, MUTED = "#e6edf7", "#93a3bd"
CYAN, PINK, ORANGE, GREEN = "#38bdf8", "#f472b6", "#f59e0b", "#34d399"
FONT = "font-family=\"'Segoe UI','PingFang SC','Microsoft YaHei','Noto Sans CJK SC',Helvetica,Arial,sans-serif\""
MATH = "font-family=\"'Cambria Math','STIX Two Math','Times New Roman',serif\""

T = {
    "en": dict(
        sub="Equation-only, parametric physics-informed neural network for supersonic wedge shocks",
        p1="ONE NETWORK, WHOLE PARAMETER PLANE", p2="PARAMETER-MODULATED NETWORK", p3="EQUATION-ONLY TRAINING",
        flow="supersonic inflow", wedge="wedge θ", shock="shock β(Ma, θ)",
        hyper="hyper-net", first="first layer", body="body", out="outputs",
        firsteq="W₀ + ΔW(Ma, θ)",
        loss=["per-law flux scaling", "wedge-tip corner weight", "augmented-Lagrangian wall", "4D residual-adaptive points"],
        nu="artificial viscosity → 0", no="no RH  ·  no shock position  ·  no shock sensor",
        chips=[("0.09° / 0.17°", "mean / max shock-angle error"), ("Ma 3–6 × θ 5–20°", "covered by one model"),
               ("up to 9.3×", "pressure ratio across the shock"), ("0.2%", "field error away from the shock")],
    ),
    "cn": dict(
        sub="只用方程的参数化物理信息神经网络 · 超声速楔面斜激波",
        p1="一个网络，覆盖整个参数平面", p2="参数调制网络", p3="只用方程训练",
        flow="超声速来流", wedge="楔角 θ", shock="激波 β(Ma, θ)",
        hyper="超网络", first="首层", body="主干", out="输出",
        firsteq="W₀ + ΔW(Ma, θ)",
        loss=["各守恒律按自身通量归一", "楔尖角点加权", "壁面增广拉格朗日", "四维残差自适应加点"],
        nu="人工粘性逐步 → 0", no="不给 RH  ·  不给激波位置  ·  不用激波探测器",
        chips=[("0.09° / 0.17°", "激波角误差 平均 / 最大"), ("Ma 3–6 × θ 5–20°", "一个模型全部覆盖"),
               ("最高 9.3 倍", "激波前后压比"), ("0.2%", "激波以外的场误差")],
    ),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, color=TXT, weight=400, anchor="start", font=FONT, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {font} font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def hero(lang):
    t = T[lang]
    W, H = 1280, 560
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         "<defs>",
         f'<linearGradient id="sh" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="{PINK}"/></linearGradient>',
         f'<linearGradient id="post" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{PINK}" stop-opacity="0.35"/><stop offset="1" stop-color="{CYAN}" stop-opacity="0.08"/></linearGradient>',
         f'<linearGradient id="nu" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{ORANGE}"/><stop offset="1" stop-color="{ORANGE}" stop-opacity="0.15"/></linearGradient>',
         f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>',
         f'<marker id="arc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{CYAN}"/></marker>',
         "</defs>",
         f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>']
    # title
    o.append(text(48, 66, "VanillaPINN", 40, TXT, 700))
    o.append(text(50, 98, t["sub"], 17, MUTED))
    # panels
    px = [(40, 400), (460, 380), (860, 380)]
    for (x0, w), title in zip(px, [t["p1"], t["p2"], t["p3"]]):
        o.append(f'<rect x="{x0}" y="124" width="{w}" height="300" rx="14" fill="{PANEL}" stroke="{LINE}"/>')
        o.append(text(x0 + 20, 152, title, 12.5, CYAN, 600, extra='letter-spacing="1.2"'))
    # ---- panel 1: wedge with a family of shocks
    ax, ay, L = 130, 395, 280
    th = math.radians(12)
    wy = ay - L * math.tan(th)
    be_main, _ = oblique_shock(4.5, th)
    sy = ay - L * math.tan(be_main)
    o.append(f'<polygon points="{ax},{ay} {ax + L},{sy:.1f} {ax + L},{wy:.1f}" fill="url(#post)"/>')
    fam = [(6, 5), (5, 10), (4, 15), (3.5, 18), (3, 20), (6, 15), (3, 10)]
    for i, (m, d) in enumerate(sorted(fam, key=lambda c: oblique_shock(c[0], math.radians(c[1]))[0])):
        b, _ = oblique_shock(m, math.radians(d))
        o.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + L}" y2="{ay - L * math.tan(b):.1f}" stroke="url(#sh)" '
                 f'stroke-width="1.3" stroke-opacity="{0.25 + 0.08 * i:.2f}"/>')
    o.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + L}" y2="{sy:.1f}" stroke="url(#sh)" stroke-width="3.2"/>')
    o.append(f'<polygon points="{ax},{ay} {ax + L},{wy:.1f} {ax + L},{ay}" fill="#33415c" stroke="#5b6b8c"/>')
    o.append(f'<line x1="60" y1="{ay}" x2="{ax + L}" y2="{ay}" stroke="#5b6b8c" stroke-dasharray="4 4"/>')
    for yy in (200, 255, 310, 365):
        o.append(f'<line x1="62" y1="{yy}" x2="112" y2="{yy}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ar)"/>')
    o.append(text(62, 186, t["flow"], 12.5, MUTED))
    o.append(text(62, 418 - 2, "Ma∞ 3–6", 13, MUTED))
    o.append(text(ax + L - 6, wy + 30, t["wedge"], 13, TXT, 600, "end"))
    o.append(text(ax + 150, sy + 28 - 0.55 * (ay - sy) * 0.0, "", 1))
    o.append(text(ax + L - 4, sy - 12, t["shock"], 13.5, PINK, 600, "end"))
    # ---- panel 2: network
    def node(x, y, r=9, fill=PANEL, stroke=CYAN):
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
    inx, hx = 500, 560
    xin = [(inx, 200, "x"), (inx, 240, "y")]
    pin = [(inx, 330, "Ma"), (inx, 370, "θ")]
    l1x, l2x, l3x, outx = 640, 705, 765, 815
    ly = [195, 230, 265, 300, 335]
    # edges
    for _, yy, _ in xin:
        for y2 in ly:
            o.append(f'<line x1="{inx}" y1="{yy}" x2="{l1x}" y2="{y2}" stroke="{LINE}" stroke-width="1"/>')
    for a, b in ((l1x, l2x), (l2x, l3x)):
        for y1 in ly:
            for y2 in ly:
                o.append(f'<line x1="{a}" y1="{y1}" x2="{b}" y2="{y2}" stroke="{LINE}" stroke-width="0.8"/>')
    outs = [(outx, 205, "u"), (outx, 245, "v"), (outx, 285, "ρ"), (outx, 325, "p")]
    for y1 in ly:
        for _, y2, _ in outs:
            o.append(f'<line x1="{l3x}" y1="{y1}" x2="{outx}" y2="{y2}" stroke="{LINE}" stroke-width="0.8"/>')
    # hyper-net
    o.append(f'<rect x="{hx - 8}" y="318" width="64" height="64" rx="10" fill="#2a1f0b" stroke="{ORANGE}" stroke-width="1.6"/>')
    o.append(text(hx + 24, 355, t["hyper"], 12, ORANGE, 600, "middle"))
    for _, yy, _ in pin:
        o.append(f'<line x1="{inx + 9}" y1="{yy}" x2="{hx - 8}" y2="{yy}" stroke="{ORANGE}" stroke-width="1.4"/>')
    o.append(f'<path d="M{hx + 56},{350} C{l1x - 20},{350} {l1x},{330} {l1x},{352}" fill="none" stroke="{ORANGE}" '
             f'stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#ar)"/>')
    o.append(f'<line x1="{l2x}" y1="352" x2="{l2x}" y2="347" stroke="{ORANGE}" stroke-width="1.4"/>')
    for x in (l1x, l2x, l3x):
        for y in ly:
            o.append(node(x, y, 8, PANEL, ORANGE if x == l1x else CYAN))
    for x, y, s in xin + pin:
        o.append(node(x, y, 13, PANEL, CYAN if s in "xy" else ORANGE))
        o.append(text(x, y + 5, s, 13, TXT, 600, "middle"))
    for x, y, s in outs:
        o.append(node(x, y, 13, PANEL, PINK))
        o.append(text(x, y + 5, s, 13, TXT, 600, "middle"))
    o.append(text(l1x, 182, t["first"], 11.5, ORANGE, 600, "middle"))
    o.append(text((l2x + l3x) / 2, 182, t["body"], 11.5, MUTED, 600, "middle"))
    o.append(text(outx, 182, t["out"], 11.5, PINK, 600, "middle"))
    o.append(text(l1x + 10, 372, t["firsteq"], 13, ORANGE, 400, "start", MATH))
    o.append(text(650, 408, "neurons rotate with the shock surface" if lang == "en" else "神经元随激波面一起转动", 12, MUTED, 400, "middle"))
    # ---- panel 3: loss
    x0 = 880
    o.append(text(x0 + 170, 196, "∇·F(U) − ν ΔU = 0", 24, TXT, 400, "middle", MATH))
    for i, s in enumerate(t["loss"]):
        y = 232 + 27 * i
        o.append(f'<circle cx="{x0 + 18}" cy="{y - 5}" r="3.5" fill="{CYAN}"/>')
        o.append(text(x0 + 32, y, s, 14, TXT))
    o.append(f'<rect x="{x0 + 14}" y="350" width="310" height="12" rx="6" fill="url(#nu)"/>')
    o.append(text(x0 + 14, 342, t["nu"], 12.5, ORANGE, 600))
    o.append(text(x0 + 14, 380, "4e-3", 11, MUTED))
    o.append(text(x0 + 324, 380, "7e-4", 11, MUTED, 400, "end"))
    o.append(text(x0 + 170, 408, t["no"], 12.5, GREEN, 600, "middle"))
    # ---- chips
    cw, gap, cx0 = 290, 13.3, 40
    for i, (big, small) in enumerate(t["chips"]):
        x = cx0 + i * (cw + gap)
        o.append(f'<rect x="{x:.1f}" y="446" width="{cw}" height="84" rx="12" fill="{PANEL}" stroke="{LINE}"/>')
        o.append(f'<rect x="{x:.1f}" y="446" width="4" height="84" rx="2" fill="{[CYAN, PINK, ORANGE, GREEN][i]}"/>')
        o.append(text(x + 22, 484, big, 23, TXT, 700))
        o.append(text(x + 22, 510, small, 13, MUTED))
    o.append("</svg>")
    return "\n".join(o)


def compare(lang):
    en = lang == "en"
    _, p_w = oblique_shock(3.0, math.radians(10))
    rows = [
        (("Wassing et al. 2024", "Comput. Fluids") if en else ("Wassing 等 2024", "Comput. Fluids"),
         (2, 3), (10, 10), p_w[3], "< 1°", "0.5–0.7%", "adaptive viscosity" if en else "自适应粘性", MUTED),
        (("Tian et al. 2026", "CJTAM") if en else ("Tian 等 2026", "力学学报"),
         (2, 3), None, None, "< 0.2°", "—", "shock-detector viscosity" if en else "激波探测器引导粘性", MUTED),
        (("This repository", "models/param_B1_w192") if en else ("本仓库", "models/param_B1_w192"),
         (3, 6), (5, 20), 9.25, "0.09° / 0.17°", "0.27%", "equations only" if en else "只用方程", CYAN),
    ]
    W, H = 1280, 400
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>',
         text(40, 52, "Parametric wedge-shock PINNs: coverage and accuracy" if en else "参数化楔面激波 PINN：覆盖范围与精度对比", 22, TXT, 700)]
    cols = [(40, "Work" if en else "工作"), (300, "Mach range" if en else "马赫数范围"), (560, "Wedge angle" if en else "楔角范围"),
            (800, "Pressure ratio" if en else "最大压比"), (990, "β error" if en else "激波角误差"),
            (1130, "Field error*" if en else "场误差*")]
    for x, s in cols:
        o.append(text(x, 96, s, 13, MUTED, 600))
    o.append(f'<line x1="40" y1="110" x2="{W - 40}" y2="110" stroke="{LINE}"/>')

    def bar(x, w, lo, hi, a, b, y, color, label):
        s = [f'<rect x="{x}" y="{y - 7}" width="{w}" height="14" rx="7" fill="{PANEL}" stroke="{LINE}"/>']
        if a is not None:
            xa, xb = x + w * (a - lo) / (hi - lo), x + w * (b - lo) / (hi - lo)
            if xb - xa < 6:
                s.append(f'<circle cx="{xa:.1f}" cy="{y}" r="7" fill="{color}"/>')
            else:
                s.append(f'<rect x="{xa:.1f}" y="{y - 7}" width="{xb - xa:.1f}" height="14" rx="7" fill="{color}"/>')
        s.append(text(x, y + 26, label, 11.5, MUTED))
        return "\n".join(s)

    for i, (name, ma, thr, pr, be, fe, how, color) in enumerate(rows):
        y = 150 + 78 * i
        if i == 2:
            o.append(f'<rect x="28" y="{y - 30}" width="{W - 56}" height="74" rx="12" fill="#0f2236" stroke="{CYAN}" stroke-opacity="0.6"/>')
        o.append(text(40, y - 2, name[0], 16, CYAN if i == 2 else TXT, 700))
        o.append(text(40, y + 20, name[1] + "  ·  " + how, 12, MUTED))
        o.append(bar(300, 220, 2, 6, ma[0], ma[1], y, color, f"Ma {ma[0]}–{ma[1]}  (axis 2–6)" if en else f"Ma {ma[0]}–{ma[1]}（坐标 2–6）"))
        lab = ("fixed, not stated" if en else "固定，未注明") if thr is None else (
            (f"θ = {thr[0]}° fixed" if en else f"θ = {thr[0]}° 固定") if thr[0] == thr[1] else f"θ {thr[0]}–{thr[1]}°  ({'axis' if en else '坐标'} 0–20°)")
        o.append(bar(560, 200, 0, 20, None if thr is None else thr[0], None if thr is None else thr[1], y, color, lab))
        if pr is None:
            o.append(text(800, y + 5, "—", 16, MUTED))
        else:
            wpr = 150 * pr / 9.5
            o.append(f'<rect x="800" y="{y - 7}" width="150" height="14" rx="7" fill="{PANEL}" stroke="{LINE}"/>')
            o.append(f'<rect x="800" y="{y - 7}" width="{wpr:.1f}" height="14" rx="7" fill="{color}"/>')
            o.append(text(800, y + 26, f"≤ {pr:.2f}".rstrip("0"), 11.5, MUTED))
        o.append(text(990, y + 6, be, 17, CYAN if i == 2 else TXT, 700))
        o.append(text(1130, y + 6, fe, 17, CYAN if i == 2 else TXT, 700))
    o.append(text(40, H - 28, ("* mean pointwise relative error of the fields; this repository: ρ 0.27 %, p 0.49 % over 16 cases (relative L2: ρ 3.0 %, p 4.4 %). Literature values from the papers; metric definitions may differ slightly."
                               if en else "* 场量逐点平均相对误差；本仓库为 16 个工况的平均：ρ 0.27%，p 0.49%（相对 L2：ρ 3.0%，p 4.4%）。文献数字取自原文，指标定义可能略有差异。"), 12, MUTED))
    o.append("</svg>")
    return "\n".join(o)


for lang in ("en", "cn"):
    open(os.path.join(ROOT, "assets", f"hero_{lang}.svg"), "w", encoding="utf-8").write(hero(lang))
    open(os.path.join(ROOT, "assets", f"compare_{lang}.svg"), "w", encoding="utf-8").write(compare(lang))
print("ok")
