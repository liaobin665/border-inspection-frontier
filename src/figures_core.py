"""Role figures for the dual-objective paper (v2 layout).

Fig 1  institutional gap  (3 panels): where the shipment count and the animals sit
Fig 2  two lists          (4 panels): capture curves, four blocks, composition, years
Fig 3  frontier           (2 panels): lambda path, budget for 80% capture
Fig 4  defence matrix     (2 panels): every robustness check, both metrics
Fig 5  purpose boundary   (2 panels): refusal gradient, the two rules under T and P

Colour rule across every panel: shipment rule = blue, animal rule = orange.
All numbers come from results/p6_*.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"

mpl.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

CASE, ANIMAL = "#0072B2", "#D55E00"
INK, GREY, LGREY = "#1A1A1A", "#8C8C8C", "#D9D9D9"
BAND_COLORS = ["#E0E0E0", "#9E9E9E", "#616161", "#262626"]
BAND_INK = ["#1A1A1A", "#1A1A1A", "#FFFFFF", "#FFFFFF"]
CM = 1 / 2.54


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def panel(ax, letter, dx=-0.14, dy=1.07):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10.0,
            fontweight="bold", va="top", ha="left", color=INK)


def save(fig, name):
    """600 dpi raster for the manuscript + vector PDF for submission."""
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    fig.savefig(FIG / name.replace(".png", ".pdf"), bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", name, "+ pdf")


# --------------------------------------------------------------------- Fig 1
def fig_gap():
    figd = load("p6_figdata.json")
    main = load("p6_main.json")
    bands = figd["T_test_bands"]["bands"]
    fig = plt.figure(figsize=(16 * CM, 12.8 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[0.55, 1.0],
                          wspace=0.30, hspace=0.52,
                          left=0.095, right=0.985, top=0.94, bottom=0.10)
    ax = fig.add_subplot(gs[0, :])

    rows = [
        ("申报行", [b["share_rows"] for b in bands]),
        ("拒收行", [b["share_refu"] for b in bands]),
        ("被拒动物", [b["share_animals"] for b in bands]),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (name, vals) in zip(ys, rows):
        left = 0.0
        for i, v in enumerate(vals):
            ax.barh(y, v * 100, left=left, height=0.50, color=BAND_COLORS[i],
                    edgecolor="white", linewidth=0.8)
            if v >= 0.030:
                ax.text(left + v * 100 / 2, y, f"{v*100:.1f}",
                        ha="center", va="center", fontsize=6.4,
                        color=BAND_INK[i])
            left += v * 100
    ax.annotate(f"{rows[1][1][-1]*100:.1f}%", xy=(100, ys[1]),
                xytext=(102.4, ys[1]), fontsize=7.0, color=INK,
                va="center", ha="left", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY,
                                shrinkA=0, shrinkB=0))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("占该类别的比例（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=c, edgecolor="white", label=l)
               for c, l in zip(BAND_COLORS, ["1–10 件", "11–100 件", "101–1000 件", ">1000 件"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=4, frameon=False, handlelength=1.1, columnspacing=1.6,
              fontsize=6.4)
    panel(ax, "a", dx=-0.055, dy=1.06)

    # (b) concentration of refused animals over refused rows
    ax = fig.add_subplot(gs[1, 0])
    con = figd["T_test_concentration"]
    x = np.array(con["curve"]["share_rows"]) * 100
    y = np.array(con["curve"]["share_animals"]) * 100
    ax.plot(x, y, color=ANIMAL, lw=1.4)
    ax.fill_between(x, 0, y, color=ANIMAL, alpha=0.10, lw=0)
    xx = np.geomspace(x.min(), 100, 240)
    ax.plot(xx, xx, ls=(0, (3, 2)), lw=0.7, color=GREY, zorder=1)
    for rank, key, off, ha in ((1, "top1", (7, -12), "left"),
                               (10, "top10", (7, -12), "left"),
                               (100, "top100", (-6, 9), "right")):
        ax.plot(x[rank - 1], y[rank - 1] * 100, "o", ms=3.2, color=ANIMAL,
                zorder=4)
        lab = f"第 {rank} 行 {con[key]*100:.1f}%"
        if rank == 1:
            lab += f"\n一票 {main["concentration"]["max_row"] / 1e4:.1f} 万只寄生蜂"
        ax.annotate(lab, xy=(x[rank - 1], con[key] * 100),
                    xytext=off, textcoords="offset points",
                    fontsize=6.6, color=INK, ha=ha, va="center")
    ax.axhline(50, color=GREY, lw=0.5, ls=":")
    ax.text(0.055, 52, "50%", fontsize=6, color=GREY, ha="left")
    ax.set_xscale("log")
    ax.set_xlim(x.min() * 0.85, 100)
    ax.set_ylim(0, 102)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("被拒行按申报量从大到小的累计比例（对数轴）")
    ax.set_ylabel("被拒动物累计比例（%）")
    ax.text(0.04, 0.06, "0.04% 的行承担 50% 的被拒动物", transform=ax.transAxes,
            fontsize=6.4, color=GREY)
    panel(ax, "b")

    # (c) refusal rate by size
    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(bands))
    ys = [b["rate"] * 100 for b in bands]
    ax.plot(xs, ys, "-o", ms=3.2, lw=1.2, color=CASE)
    for x, b, y in zip(xs, bands, ys):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.2, color=CASE)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}\nn={int(r['refu'])}"
                        for b, r in zip(["1–10", "11–100", "101–1000", ">1000"], bands)])
    ax.set_xlabel("申报量（件）")
    ax.set_ylabel("拒收率（%）")
    ax.set_ylim(0, 1.95)
    ax.set_xlim(-0.45, 3.45)
    panel(ax, "c")

    save(fig, "fig1_institutional_gap.png")


# --------------------------------------------------------------------- Fig 2
def fig_lists():
    main, roll = load("p6_main.json"), load("p6_rolling.json")
    figd = load("p6_figdata.json")
    rev = load("p6_revisions.json")
    fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 13.6 * CM))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.10,
                        wspace=0.34, hspace=0.46)

    # (a) capture against budget
    ax = axes[0, 0]
    cur = main["sweep"]["curve"]
    b = np.array([r["budget"] for r in cur]) * 100
    me = [i for i, bv in enumerate(b) if round(bv) in (10, 20, 30, 40, 50)]
    ax.plot(b, [r["p_cap_refu"] * 100 for r in cur], "-", lw=1.3, color=CASE,
            marker="o", ms=2.4, markevery=me)
    ax.plot(b, [r["p_cap_spec"] * 100 for r in cur], "--", lw=1.1, color=CASE,
            marker="s", ms=2.2, markevery=me)
    ax.plot(b, [r["pq_cap_refu"] * 100 for r in cur], "-", lw=1.3, color=ANIMAL,
            marker="o", ms=2.4, markevery=me)
    ax.plot(b, [r["pq_cap_spec"] * 100 for r in cur], "--", lw=1.1, color=ANIMAL,
            marker="s", ms=2.2, markevery=me)
    qs = rev["X1_quantity_only"]["sweep"]
    ax.plot([r["budget"] * 100 for r in qs], [r["q_cap_spec"] * 100 for r in qs],
            ls=(0, (1, 1.4)), lw=1.0, color=GREY, marker="^", ms=2.0,
            markevery=[i for i, bv in enumerate(r["budget"] * 100 for r in qs)
                       if round(float(bv)) in (10, 20, 30, 40, 50)])
    bp = main["bootstrap"]["point"]
    ax.axvline(10, color=GREY, lw=0.6, ls=":")
    ax.text(1.2, 12.0, f"票口径 {bp['p_cap_refu']*100:.1f} / {bp['p_cap_spec']*100:.1f}",
            fontsize=6.2, color=CASE, va="center")
    ax.text(1.2, 5.0, f"动物口径 {bp['pq_cap_refu']*100:.1f} / {bp['pq_cap_spec']*100:.1f}",
            fontsize=6.2, color=ANIMAL, va="center")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("检查预算（占申报行比例，%）")
    ax.set_ylabel("捕获率（%）")
    ax.text(9.4, 96, "10% 预算", fontsize=7.0, color=GREY, va="top", ha="right")
    handles = [Line2D([], [], color=CASE, lw=1.4, label="票口径（行级）"),
               Line2D([], [], color=ANIMAL, lw=1.4, label="动物口径"),
               Line2D([], [], color=GREY, lw=1.0, ls=(0, (1, 1.4)),
                      label="纯申报量")]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handlelength=1.5, labelspacing=0.22, borderaxespad=0.25)
    panel(ax, "a")

    # (b) four blocks
    ax = axes[0, 1]
    det = main["composition"]["detail"]
    refu, anim = main["refu"], main["refused_animals"]
    sh = det["refused_rows_shared"]
    cs = [sh, det["refused_rows_p"] - sh, det["refused_rows_pq"] - sh,
          refu - det["refused_rows_p"] - det["refused_rows_pq"] + sh]
    ash = det["animals_shared"]
    an = [ash, det["animals_p_only"], det["animals_pq_only"],
          anim - ash - det["animals_p_only"] - det["animals_pq_only"]]
    spec = [("票口径（行级）", [c / refu for c in cs]), ("被拒动物", [a / anim for a in an])]
    cols = [GREY, CASE, ANIMAL, "#F0F0F0"]
    ys = [1, 0]
    for y, (name, vals) in zip(ys, spec):
        left = 0.0
        for i, v in enumerate(vals):
            ax.barh(y, v * 100, left=left, height=0.5, color=cols[i],
                    edgecolor="white", linewidth=0.8,
                    hatch="///" if i == 3 else None)
            if v > 0.06:
                ax.text(left + v * 100 / 2, y, f"{v*100:.1f}",
                        ha="center", va="center", fontsize=6.4,
                        color="white" if i in (1, 2) else INK)
            left += v * 100
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in spec])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("占全部拒收结果的比例（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=cols[0], label="两份都抓到"),
               Patch(facecolor=cols[1], label="只被票口径名单"),
               Patch(facecolor=cols[2], label="只被动物口径名单"),
               Patch(facecolor=cols[3], hatch="///", edgecolor=GREY,
                     label="两份都漏掉")]
    ax.set_ylim(-0.55, 2.05)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, handlelength=1.0, columnspacing=1.2,
              fontsize=6.2)
    panel(ax, "b")

    # (c) composition of the two lists
    ax = axes[1, 0]
    comp = main["composition"]
    cols_p = [comp["p"][f"share_band{i}"] for i in range(1, 5)]
    cols_q = [comp["pq"][f"share_band{i}"] for i in range(1, 5)]
    for y, vals, name in ((1, cols_p, "票口径名单"), (0, cols_q, "动物口径名单")):
        left = 0.0
        for i, v in enumerate(vals):
            ax.barh(y, v * 100, left=left, height=0.5, color=BAND_COLORS[i],
                    edgecolor="white", linewidth=0.8)
            if v >= 0.055:
                ax.text(left + v * 100 / 2, y, f"{v*100:.1f}", ha="center",
                        va="center", fontsize=6.4,
                        color=BAND_INK[i])
            left += v * 100
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["票口径名单", "动物口径名单"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("名单中申报行的档位构成（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel(ax, "c")

    # (d) rolling years
    ax = axes[1, 1]
    years = sorted(roll)
    dc = [(roll[y]["at10"]["p_cap_refu"] - roll[y]["at10"]["pq_cap_refu"]) * 100
          for y in years]
    da = [(roll[y]["at10"]["pq_cap_spec"] - roll[y]["at10"]["p_cap_spec"]) * 100
          for y in years]
    yy = np.arange(len(years))
    for i, y in enumerate(years):
        ax.plot([dc[i], da[i]], [i, i], color=LGREY, lw=0.8, zorder=1)
    ax.plot(dc, yy, "o", ms=3.0, color=CASE, zorder=3, label="票口径捕获：票口径名单多抓")
    ax.plot(da, yy, "o", ms=3.0, color=ANIMAL, zorder=3, label="动物口径捕获：动物口径名单多抓")
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([str(y) for y in years])
    ax.set_ylim(-0.7, len(years) - 0.3)
    ax.set_xlim(-8, 88)
    ax.set_xlabel("捕获率差值（百分点）")
    ax.legend(loc="upper center", frameon=False, handlelength=1.2,
              labelspacing=0.25, bbox_to_anchor=(0.5, -0.20), ncol=2,
              columnspacing=1.6)
    panel(ax, "d")

    save(fig, "fig2_dual_objective.png")


# --------------------------------------------------------------------- Fig 3
def fig_frontier():
    main = load("p6_main.json")
    rev = load("p6_revisions.json")
    sw = main["sweep"]
    x2 = rev["X2_cost"]
    fig = plt.figure(figsize=(16 * CM, 7.4 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.92, 0.86], wspace=0.44,
                          left=0.075, right=0.985, top=0.92, bottom=0.15)

    ax = fig.add_subplot(gs[0, 0])
    lam = main["lambda"]["lam"]
    xs = [r["cap_refu"] * 100 for r in lam]
    ys = [r["cap_spec"] * 100 for r in lam]
    ax.plot(xs, ys, "-", color=GREY, lw=0.9, zorder=2)
    ax.scatter(xs, ys, s=9, color=GREY, zorder=3)
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], ls=(0, (3, 2)), lw=0.8,
            color=LGREY, zorder=1)
    for idx, lab, col in ((0, "λ=0", CASE), (-1, "λ=1", ANIMAL)):
        ax.scatter([xs[idx]], [ys[idx]], s=34, color=col, zorder=4)
        ax.annotate(lab, (xs[idx], ys[idx]), xytext=(6, -12) if idx == 0 else (-4, -16),
                    textcoords="offset points", fontsize=7, color=col,
                    fontweight="bold")
    mid = next(r for r in lam if r["lam"] == 0.5)
    ax.scatter([mid["cap_refu"] * 100], [mid["cap_spec"] * 100], s=34,
               color="#009E73", zorder=4)
    ax.annotate("λ=0.5\n64.4 / 78.3", (mid["cap_refu"] * 100, mid["cap_spec"] * 100),
                xytext=(11, 9), textcoords="offset points", fontsize=6.6,
                color="#009E73")
    ax.set_xlim(38, 74)
    ax.set_ylim(40, 94)
    ax.set_xlabel("拒收票捕获（%）")
    ax.set_ylabel("被拒动物捕获（%）")
    ax.text(0.03, 0.05, "10% 预算", transform=ax.transAxes, fontsize=6.4,
            color=GREY, va="bottom")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    groups = [("80% 动物", sw["budget_80spec_p"] * 100,
               sw["budget_80spec_pq"] * 100,
               sw["budget_80spec_p"] / sw["budget_80spec_pq"]),
              ("80% 票", sw["budget_80refu_p"] * 100,
               sw["budget_80refu_pq"] * 100,
               sw["budget_80refu_pq"] / sw["budget_80refu_p"])]
    ys = [1, 0]
    for y, (name, vp, vq, ratio) in zip(ys, groups):
        ax.barh(y + 0.16, vp, height=0.26, color=CASE)
        ax.barh(y - 0.16, vq, height=0.26, color=ANIMAL)
        ax.text(vp + 1.2, y + 0.16, f"{vp:.1f}%", va="center", fontsize=6.4,
                color=CASE)
        ax.text(vq + 1.2, y - 0.16, f"{vq:.1f}%", va="center", fontsize=6.4,
                color=ANIMAL)
        ax.text(52, y, f"{ratio:.1f}×", va="center", fontsize=7.4,
                fontweight="bold", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("达到目标所需的检查预算（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=CASE, label="票口径排序"),
               Patch(facecolor=ANIMAL, label="动物口径排序")]
    ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.0,
              borderaxespad=0.15)
    panel(ax, "b")

    # (c) how the 80%-animal budget ratio moves with the cost exponent
    ax = fig.add_subplot(gs[0, 2])
    sw = x2["sweep"]
    gx = np.array([r["gamma"] for r in sw])
    gy = np.array([r["ratio_80anim"] for r in sw])
    ax.axhspan(1.0, 7.6, color=CASE, alpha=0.055, lw=0)
    ax.axhspan(0.30, 1.0, color=ANIMAL, alpha=0.07, lw=0)
    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.plot(gx, gy, "-", color=INK, lw=1.3, zorder=3)
    ax.plot(gx, gy, "o", ms=2.4, color=INK, zorder=4)
    cross = float(np.interp(1.0, gy[::-1], gx[::-1]))
    for g, lab, off, ha in ((0.0, f"{main['sweep']['budget_80spec_p'] / main['sweep']['budget_80spec_pq']:.1f}×",
                             (8, 3), "left"),
                            (0.5, f"{x2['gamma0.5']['ratio_80anim']:.1f}×", (7, 6), "left"),
                            (1.0, f"{x2['gamma1.0']['ratio_80anim']:.1f}×", (8, -2), "left")):
        v = float(np.interp(g, gx, gy))
        ax.plot(g, v, "o", ms=4.4, color=CASE if v >= 1 else ANIMAL,
                mec="white", mew=0.6, zorder=5)
        ax.annotate(lab, (g, v), xytext=off, textcoords="offset points",
                    fontsize=6.8, fontweight="bold", color=INK, ha=ha,
                    annotation_clip=False)
    ax.plot(cross, 1.0, "D", ms=3.6, color=INK, zorder=6)
    ax.text(0.98, 7.35, "上：动物排序更省", fontsize=6.0, color=CASE, va="top", ha="right")
    ax.text(0.02, 0.22, "下：动物排序更贵", fontsize=6.0, color=ANIMAL, va="bottom", ha="left")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim(-0.03, 1.16)
    ax.set_ylim(0.20, 7.6)
    ax.set_yticks([0.5, 1, 2, 4, 6])
    ax.set_xlabel("成本指数 γ（单行成本 ∝ $q^{γ}$）")
    ax.set_ylabel("80% 动物捕获的预算倍数", labelpad=2)
    panel(ax, "c")

    save(fig, "fig3_frontier.png")


# --------------------------------------------------------------------- Fig 4
def fig_defence():
    main = load("p6_main.json")
    p5e = load("p5e_whole_animals.json")
    alt, ship, purpose = main["alt_score"], main["shipment"], load("p6_purpose_p.json")
    cap = p5e["all_terms"]["metrics"]
    rows = [
        ("主结果（封顶 1e4）", main["bootstrap"]["point"]["p_cap_refu"],
         main["bootstrap"]["point"]["p_cap_spec"],
         main["bootstrap"]["point"]["pq_cap_refu"],
         main["bootstrap"]["point"]["pq_cap_spec"]),
        ("原始", cap["raw"]["p"]["10%"]["cap_refu"],
         cap["raw"]["p"]["10%"]["cap_spec"],
         cap["raw"]["pq"]["10%"]["cap_refu"],
         cap["raw"]["pq"]["10%"]["cap_spec"]),
        ("封顶 1e3", cap["cap_1000"]["p"]["10%"]["cap_refu"],
         cap["cap_1000"]["p"]["10%"]["cap_spec"],
         cap["cap_1000"]["pq"]["10%"]["cap_refu"],
         cap["cap_1000"]["pq"]["10%"]["cap_spec"]),
        ("替代：历史物种率", alt["hist_sp"]["p_cap_refu"], alt["hist_sp"]["p_cap_spec"],
         alt["hist_sp"]["pq_cap_refu"], alt["hist_sp"]["pq_cap_spec"]),
        ("替代：历史物种×原产国", alt["hist_sporg"]["p_cap_refu"],
         alt["hist_sporg"]["p_cap_spec"], alt["hist_sporg"]["pq_cap_refu"],
         alt["hist_sporg"]["pq_cap_spec"]),
        ("替代：历史口岸", alt["hist_port"]["p_cap_refu"],
         alt["hist_port"]["p_cap_spec"], alt["hist_port"]["pq_cap_refu"],
         alt["hist_port"]["pq_cap_spec"]),
        ("整票复核（2019–2020，真票）", ship["p_any"]["p_cap_refu"],
         ship["p_any"]["p_cap_spec"], ship["p_qty"]["p_cap_refu"],
         ship["p_qty"]["p_cap_spec"]),
        ("用途边界：个人进口", purpose["at10"]["p_cap_refu"],
         purpose["at10"]["p_cap_spec"], purpose["at10"]["pq_cap_refu"],
         purpose["at10"]["pq_cap_spec"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 8.6 * CM), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.16, wspace=0.10)
    ys = np.arange(len(rows))[::-1]
    for ax, key, title in ((axes[0], 2, "票口径捕获（%）"), (axes[1], 3, "被拒动物捕获（%）")):
        for y, r in zip(ys, rows):
            a, b = r[1] * 100, r[3] * 100
            if key == 3:
                a, b = r[2] * 100, r[4] * 100
            ax.plot([a, b], [y, y], color=LGREY, lw=1.6, zorder=1,
                    solid_capstyle="round")
            ax.plot(a, y, "o", ms=3.4, color=CASE, zorder=3)
            ax.plot(b, y, "o", ms=3.4, color=ANIMAL, zorder=3)
            # put each label outside the pair, so short gaps never collide
            if b >= a:
                bx, bh = (b + 2.2, "left") if b < 94 else (b - 2.2, "right")
                axx, ah = (a - 2.2, "right") if a > 6 else (a + 2.2, "left")
            else:
                bx, bh = (b - 2.2, "right") if b > 6 else (b + 2.2, "left")
                axx, ah = (a + 2.2, "left") if a < 94 else (a - 2.2, "right")
            ax.text(bx, y, f"{b:.0f}", va="center", ha=bh, fontsize=6.0, color=ANIMAL)
            ax.text(axx, y, f"{a:.0f}", va="center", ha=ah, fontsize=6.0, color=CASE)
        ax.set_xlim(0, 100)
        ax.set_xlabel(title)
        ax.set_ylim(-0.65, len(rows) - 0.35)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=CASE,
                      label="票口径排序"),
               Line2D([], [], marker="o", ls="", ms=4, color=ANIMAL,
                      label="动物口径排序")]
    axes[1].legend(handles=handles, loc="lower center", frameon=False,
                   handletextpad=0.3, borderaxespad=0.2, ncol=2,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    panel(axes[0], "a", dx=-0.33)
    panel(axes[1], "b", dx=-0.06)
    save(fig, "fig4_defence.png")


# --------------------------------------------------------------------- Fig 5
def fig_purpose():
    figd = load("p6_figdata.json")
    purpose = load("p6_purpose_p.json")
    main = load("p6_main.json")
    fig = plt.figure(figsize=(16 * CM, 7.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32,
                          left=0.085, right=0.985, top=0.92, bottom=0.15)

    ax = fig.add_subplot(gs[0, 0])
    # log axis: the commercial gradient (0.3-1.4%) and the personal one (20-30%)
    # must both be readable. The >1000 band of the personal window holds 5 rows
    # and no refusal, so it cannot be drawn on a log axis; it is stated instead.
    bands_t = figd["T_test_bands"]["bands"]
    bands_p = figd["P_test_bands"]["bands"]
    xt = np.arange(len(bands_t))
    yt = [b["rate"] * 100 for b in bands_t]
    xp = np.arange(3)
    yp = [b["rate"] * 100 for b in bands_p[:3]]
    ax.plot(xt, yt, "-o", ms=3.4, lw=1.3, color=CASE)
    ax.plot(xp, yp, "-o", ms=3.4, lw=1.3, color=ANIMAL)
    ax.text(0.03, 0.47, "商业进口（T）", fontsize=7.4, color=CASE,
            ha="left", va="center")
    ax.text(1.32, 52, "个人进口（P）", fontsize=7.4, color=ANIMAL,
            ha="left", va="center")
    for x, y in zip(xt, yt):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.8, color=CASE)
    for x, y in zip(xp, yp):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=6.8, color=ANIMAL)
    ax.text(3.3, 0.225, f">1000：{int(bands_p[-1]['rows'])} 行、"
            f"{int(bands_p[-1]['refu'])} 例拒收", fontsize=6.6, color=ANIMAL,
            ha="right", va="bottom")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["1–10", "11–100", "101–1000", ">1000"])
    ax.set_xlim(-0.35, 3.35)
    ax.set_xlabel("申报量（件）")
    ax.set_ylabel("拒收率（%）")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 60)
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax.minorticks_off()
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    bp = main["bootstrap"]["point"]
    pts = {
        "T 票数": (bp["p_cap_refu"], bp["p_cap_spec"], CASE, "o"),
        "T 动物": (bp["pq_cap_refu"], bp["pq_cap_spec"], CASE, "D"),
        "P 票数": (purpose["at10"]["p_cap_refu"], purpose["at10"]["p_cap_spec"], ANIMAL, "o"),
        "P 动物": (purpose["at10"]["pq_cap_refu"], purpose["at10"]["pq_cap_spec"], ANIMAL, "D"),
    }
    for purpose_key, a, b in (("T", "T 票数", "T 动物"), ("P", "P 票数", "P 动物")):
        x0, y0, c0, _ = pts[a]
        x1, y1, _, _ = pts[b]
        ax.annotate("", xy=(x1 * 100, y1 * 100), xytext=(x0 * 100, y0 * 100),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=c0,
                                    shrinkA=6, shrinkB=6))
        ax.text((x0 + x1) * 50 + 1.2, (y0 + y1) * 50, purpose_key,
                fontsize=7.0, color=c0, fontweight="bold")
    for name, (x, y, c, m) in pts.items():
        ax.plot(x * 100, y * 100, m, ms=4.6, color=c, zorder=4)
        above = m == "D"
        ax.annotate(f"{x*100:.1f} / {y*100:.1f}", (x * 100, y * 100),
                    xytext=(0, 9) if above else (7, -10),
                    textcoords="offset points", fontsize=6.0, color=c,
                    ha="center" if above else "left",
                    va="bottom" if above else "top")
    ax.set_xlim(20, 80)
    ax.set_ylim(38, 96)
    ax.set_xlabel("票口径捕获（%）")
    ax.set_ylabel("被拒动物捕获（%）")
    handles = [Line2D([], [], color=CASE, lw=1.0, label="商业进口 T"),
               Line2D([], [], color=ANIMAL, lw=1.0, label="个人进口 P"),
               Line2D([], [], marker="o", ls="", color=INK, ms=4, label="票口径排序"),
               Line2D([], [], marker="D", ls="", color=INK, ms=4, label="动物口径排序")]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    panel(ax, "b")

    save(fig, "fig5_purpose.png")


if __name__ == "__main__":
    fig_gap()
    fig_lists()
    fig_frontier()
    fig_defence()
    fig_purpose()
