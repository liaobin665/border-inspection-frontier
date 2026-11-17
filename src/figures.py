"""English editions of the six manuscript figures.

Layout, geometry and every number come from the existing Chinese builders;
this script re-runs them, translates each text artist through the rule table,
switches to Times, and writes figures/en/*.png (+ pdf).

Three panels need English-length geometry rather than a pure text swap, so
they are re-laid out here against the same JSON files:
  fig5_purpose  - legend moved below the axes (English legend overlapped the
                  69.1 / 45.2 point label);
  fig3_main     - two-line vertical labels and a wider left margin (English
                  labels are longer than the panel height / left margin);
  fig2_sankey   - canvas widened from 163 to 170 mm so the left-hand country
                  column is no longer clipped by the edge.
The copied panels keep the Chinese data calls, colours and panel letters.

Run: python scripts/make_p6_figs_en.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FIG_EN = ROOT / "figures"
sys.path.insert(0, str(HERE))

CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")

RULES: list[tuple[re.Pattern, str | object]] = [
    (re.compile(r"^申报行$"), "Declaration rows"),
    (re.compile(r"^拒收行$"), "Refused rows"),
    (re.compile(r"^被拒动物$"), "Refused animals"),
    (re.compile(r"^占该类别的比例（%）$"), "Share of each total (%)"),
    (re.compile(r"^(\d+–\d+|>\d+) 件$"), r"\1"),
    (re.compile(r"^第 (\d+) 行 ([\d.]+)%\n一票 ([\d.]+) 万只寄生蜂$"),
     lambda m: f"Row {m.group(1)}: {m.group(2)}%\none shipment: "
               f"{int(round(float(m.group(3)) * 10_000)):,} parasitoid wasps"),
    (re.compile(r"^第 (\d+) 行 ([\d.]+)%$"), r"Row \1: \2%"),
    (re.compile(r"^被拒行按申报量从大到小的累计比例（对数轴）$"),
     "Cumulative share of refused rows, largest first (log scale)"),
    (re.compile(r"^被拒动物累计比例（%）$"),
     "Cumulative share of refused animals (%)"),
    (re.compile(r"^0\.04% 的行承担 50% 的被拒动物$"),
     "0.04% of rows carry 50% of the refused animals"),
    (re.compile(r"^申报量（件）$"), "Declared quantity (items)"),
    (re.compile(r"^拒收率（%）$"), "Refusal rate (%)"),
    (re.compile(r"^检查预算（占申报行比例，%）$"),
     "Inspection budget (% of declaration rows)"),
    (re.compile(r"^捕获率（%）$"), "Capture rate (%)"),
    (re.compile(r"^票口径（行级）$"), "Shipment ranking (rows)"),
    (re.compile(r"^票口径$"), "Shipment ranking"),
    (re.compile(r"^动物口径$"), "Animal ranking"),
    (re.compile(r"^纯申报量$"), "Quantity only"),
    (re.compile(r"^票口径 ([\d.]+) / ([\d.]+)$"), r"Shipment \1 / \2"),
    (re.compile(r"^动物口径 ([\d.]+) / ([\d.]+)$"), r"Animal \1 / \2"),
    (re.compile(r"^10% 预算$"), "10% budget"),
    (re.compile(r"^占全部拒收结果的比例（%）$"),
     "Share of all refusal outcomes (%)"),
    (re.compile(r"^两份都抓到$"), "Both"),
    (re.compile(r"^只被票口径名单$"), "Shipment list only"),
    (re.compile(r"^只被动物口径名单$"), "Animal list only"),
    (re.compile(r"^两份都漏掉$"), "Neither"),
    (re.compile(r"^票口径名单$"), "Shipment list"),
    (re.compile(r"^动物口径名单$"), "Animal list"),
    (re.compile(r"^名单中申报行的档位构成（%）$"),
     "Size composition of list rows (%)"),
    (re.compile(r"^捕获率差值（百分点）$"),
     "Capture-rate difference (percentage points)"),
    (re.compile(r"^票口径捕获：票口径名单多抓$"),
     "Shipment capture: shipment list catches more"),
    (re.compile(r"^动物口径捕获：动物口径名单多抓$"),
     "Animal capture: animal list catches more"),
    (re.compile(r"^主结果（封顶 1e4）$"), "Main result (cap 10,000)"),
    (re.compile(r"^原始$"), "Raw (no cap)"),
    (re.compile(r"^封顶 1e3$"), "Cap 1,000"),
    (re.compile(r"^替代：历史物种率$"), "Alt.: historical species rate"),
    (re.compile(r"^替代：历史物种×原产国$"),
     "Alt.: historical species × origin"),
    (re.compile(r"^替代：历史口岸$"), "Alt.: historical port"),
    (re.compile(r"^整票复核（2019–2020，真票）$"),
     "Shipment-level check (2019–2020)"),
    (re.compile(r"^用途边界：个人进口$"), "Boundary: personal imports"),
    (re.compile(r"^票口径捕获（%）$"), "Shipment capture (%)"),
    (re.compile(r"^被拒动物捕获（%）$"), "Refused animals captured (%)"),
    (re.compile(r"^票口径排序$"), "Shipment ranking"),
    (re.compile(r"^动物口径排序$"), "Animal ranking"),
    (re.compile(r"^商业进口（T）$"), "Commercial (T)"),
    (re.compile(r"^个人进口（P）$"), "Personal (P)"),
    (re.compile(r"^>1000：(\d+) 行、(\d+) 例拒收$"),
     r">1000: \1 rows, \2 refusals"),
    (re.compile(r"^商业进口 T$"), "Commercial imports (T)"),
    (re.compile(r"^个人进口 P$"), "Personal imports (P)"),
    (re.compile(r"^80% 动物$"), "80% animals"),
    (re.compile(r"^80% 票口径$"), "80% shipments"),
    (re.compile(r"^票口径排序 ([\d.]+)%$"), r"Shipment ranking \1%"),
    (re.compile(r"^动物口径排序 ([\d.]+)%$"), r"Animal ranking \1%"),
    (re.compile(r"^达到目标所需的检查预算（%）$"),
     "Inspection budget needed for target (%)"),
    (re.compile(r"^倍数（票口径÷动物口径）$"),
     "Multiple\n(shipment ÷ animal)"),
    (re.compile(r"^固定成本占比（1 − γ）$"), "Fixed-cost share (1 − γ)"),
    (re.compile(r"^动物口径更省$"), "Animal ranking cheaper"),
    (re.compile(r"^动物口径更贵$"), "Animal ranking costlier"),
    (re.compile(r"^80% 目标$"), "80% target"),
    (re.compile(r"^穿越点 γ = ([\d.]+)–([\d.]+)$"), r"Crossing γ = \1–\2"),
    (re.compile(r"^(\d+)% 动物捕获$"), r"\1% animal capture"),
    (re.compile(r"^成本指数 γ（单行检查成本 ∝ \$q\^\{γ\}\$）$"),
     lambda m: "Cost index γ (per-row cost $\\propto q^{γ}$)"),
    (re.compile(r"^票口径：宽度＝拒收行数$"),
     "Shipment ranking: width = refused rows"),
    (re.compile(r"^([\d,]+) 条拒收行 · (\d+) 个口岸$"),
     r"\1 refused rows · \2 ports"),
    (re.compile(r"^>1000 档只占 ([\d.]+)% 的行$"), r">1000: only \1% of rows"),
    (re.compile(r"^动物口径：宽度＝被拒动物数$"),
     "Animal ranking: width = refused animals"),
    (re.compile(r"^([\d,]+) 只（单行封顶 1 万）$"),
     r"\1 animals (cap 10,000 per row)"),
    (re.compile(r"^>1000 档占 ([\d.]+)% 的动物$"), r">1000: \1% of animals"),
    (re.compile(r"^同一批拒收，两种计量结构$"),
     "The same refusals, two measurement structures"),
    (re.compile(r"^其他$"), "Other"),
    (re.compile(r"^1–10 件$"), "1–10 items"),
    (re.compile(r"^11–100 件$"), "11–100 items"),
    (re.compile(r"^101–1000 件$"), "101–1000 items"),
    (re.compile(r"^>1000 件$"), ">1000 items"),
]


def translate(s: str) -> str:
    for pat, rep in RULES:
        m = pat.match(s)
        if m:
            return rep(m) if callable(rep) else m.expand(rep)
    return s


def translate_fig(fig) -> list[str]:
    # Tick labels are re-rendered from the axis formatter on every draw, so a
    # set_text() on the Text artist alone is overwritten; go through the axis.
    for ax in fig.axes:
        for axis, setter in ((ax.xaxis, ax.set_xticklabels),
                             (ax.yaxis, ax.set_yticklabels)):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            new = [translate(s) for s in labels]
            if new != labels:
                setter(new)
    left = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s:
            continue
        new = translate(s)
        if new != s:
            t.set_text(new)
        if CJK.search(t.get_text()):
            left.append(t.get_text())
    return left


# --------------------------------------------------------------- panel fixes
def fix_lists(fig) -> None:
    """Set the 10% budget flag along its line: the horizontal English word is
    wider than the Chinese one and now crosses both lower curves."""
    for t in fig.axes[0].texts:
        if t.get_text() == "10% 预算" and t.get_va() == "top":
            t.set_position((9.2, 98.8))
            t.set_rotation(0)
            t.set_ha("right")
            t.set_va("top")
    # the two English value labels are wider than the Chinese ones and now run
    # through the 10% guide line; mask it behind them
    for t in fig.axes[0].texts:
        if t.get_text().startswith(("票口径 ", "动物口径 ")):
            t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6))


def fig_purpose_en(CM, CASE, ANIMAL, INK, panel) -> None:
    """Same two panels as the Chinese builder, legend moved below the axes."""
    from matplotlib.lines import Line2D

    def load(name):
        return json.loads((RES / name).read_text(encoding="utf-8"))

    figd, purpose, main = load("p6_figdata.json"), load("p6_purpose_p.json"), load("p6_main.json")
    fig = plt.figure(figsize=(16 * CM, 8.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32,
                          left=0.085, right=0.985, top=0.90, bottom=0.26)

    ax = fig.add_subplot(gs[0, 0])
    bands_t = figd["T_test_bands"]["bands"]
    bands_p = figd["P_test_bands"]["bands"]
    xt = np.arange(len(bands_t))
    yt = [b["rate"] * 100 for b in bands_t]
    xp = np.arange(3)
    yp = [b["rate"] * 100 for b in bands_p[:3]]
    ax.plot(xt, yt, "-o", ms=3.4, lw=1.3, color=CASE)
    ax.plot(xp, yp, "-o", ms=3.4, lw=1.3, color=ANIMAL)
    ax.text(0.03, 0.47, "Commercial (T)", fontsize=7.4, color=CASE,
            ha="left", va="center")
    ax.text(1.32, 52, "Personal (P)", fontsize=7.4, color=ANIMAL,
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
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, ncol=2, fontsize=6.6, handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0.2)
    panel(ax, "b")

    save(fig, "fig5_purpose.png")


def fig3_en(CM, CASE, ANIMAL, INK, GREY, LGREY, panel, ratio_axes) -> None:
    """Fig 6 of the manuscript, re-laid out for English label lengths."""
    def load(name):
        return json.loads((RES / name).read_text(encoding="utf-8"))

    main, d = load("p6_main.json"), load("p6_shadow_cost.json")
    fig = plt.figure(figsize=(18.3 * CM, 10.2 * CM))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.95,
                          left=0.115, right=0.925, top=0.92, bottom=0.17)

    ax = fig.add_subplot(gs[0, 0])
    lam = main["lambda"]["lam"]
    xs = [r["cap_refu"] * 100 for r in lam]
    ys = [r["cap_spec"] * 100 for r in lam]
    ax.plot(xs, ys, "-", color=GREY, lw=0.9, zorder=2)
    ax.scatter(xs, ys, s=8, color=GREY, zorder=3)
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], ls=(0, (3, 2)), lw=0.8,
            color=LGREY, zorder=1)
    for idx, lab, col in ((0, "λ=0", CASE), (-1, "λ=1", ANIMAL)):
        ax.scatter([xs[idx]], [ys[idx]], s=30, color=col, zorder=4)
        ax.annotate(lab, (xs[idx], ys[idx]),
                    xytext=(7, 5) if idx == 0 else (-7, 5),
                    textcoords="offset points", fontsize=7, color=col,
                    ha="left" if idx == 0 else "right")
    mid = next(r for r in lam if r["lam"] == 0.5)
    ax.scatter([mid["cap_refu"] * 100], [mid["cap_spec"] * 100], s=46,
               facecolor="white", edgecolor=INK, linewidth=1.2, zorder=5)
    ax.annotate(f"λ=0.5\n{mid['cap_refu']*100:.1f} / {mid['cap_spec']*100:.1f}",
                (mid["cap_refu"] * 100, mid["cap_spec"] * 100),
                xytext=(10, 8), textcoords="offset points", fontsize=7.0,
                color=INK)
    ax.set_xlim(38, 74)
    ax.set_ylim(40, 94)
    ax.set_xlabel("票口径捕获（%）")
    ax.set_ylabel("Refused animals\ncaptured (%)")
    ax.text(0.03, 0.05, "10% 预算", transform=ax.transAxes, fontsize=6.4,
            color=GREY, va="bottom")
    panel(ax, "a", dx=-0.22, dy=1.17)

    ax = fig.add_subplot(gs[0, 1])
    sw = main["sweep"]
    groups = [("80% 动物", sw["budget_80spec_p"] * 100,
               sw["budget_80spec_pq"] * 100,
               sw["budget_80spec_p"] / sw["budget_80spec_pq"]),
              ("80% 票口径", sw["budget_80refu_p"] * 100,
               sw["budget_80refu_pq"] * 100,
               sw["budget_80refu_pq"] / sw["budget_80refu_p"])]
    for y, (name, vp, vq, ratio) in zip([1, 0], groups):
        ax.barh(y + 0.16, vp, height=0.26, color=CASE)
        ax.barh(y - 0.16, vq, height=0.26, color=ANIMAL)
        ax.text(vp + 0.8, y + 0.16, f"票口径排序 {vp:.1f}%", va="center",
                fontsize=6.2, color=CASE)
        ax.text(vq + 0.8, y - 0.16, f"动物口径排序 {vq:.1f}%", va="center",
                fontsize=6.2, color=ANIMAL)
        ax.text(79, y, f"{ratio:.1f}×", va="center", fontsize=7.6, color=INK,
                ha="right")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("达到目标所需的检查预算（%）")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel(ax, "b", dx=-0.16, dy=1.17)

    ax = fig.add_subplot(gs[1, 0])
    rs = d["ratio80"]
    gx = np.array([r["gamma"] for r in rs])
    gy = np.array([r["ratio"] for r in rs])
    cross = d["cross_80"]["gamma"]
    ratio_axes(ax, (0.2, 9.0))
    ax.plot(gx, gy, "-", color=INK, lw=1.3, zorder=3)
    ax.plot(gx, gy, "o", ms=1.8, color=INK, zorder=4)
    for g, lab, off in ((0.0, f"{np.interp(0.0, gx, gy):.1f}×", (7, 2)),
                        (0.5, f"{np.interp(0.5, gx, gy):.2f}×", (7, 9)),
                        (1.0, f"{np.interp(1.0, gx, gy):.2f}×", (-3, 9))):
        v = float(np.interp(g, gx, gy))
        ax.plot(g, v, "o", ms=4.0, color=CASE if v >= 1 else ANIMAL,
                mec="white", mew=0.6, zorder=5)
        ax.annotate(lab, (g, v), xytext=off, textcoords="offset points",
                    fontsize=7.0, color=INK, annotation_clip=False,
                    ha="left" if off[0] > 0 else "right")
    ax.plot([cross], [1.0], "D", ms=3.6, color=INK, zorder=6)
    ax.axvline(cross, color=LGREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"γ≈{cross:.2f}", (cross, 1.0), xytext=(-26, -15),
                textcoords="offset points", fontsize=6.8, color=INK, ha="right")
    ax.text(0.97, 8.4, "动物口径更省", fontsize=6.6, color=CASE, ha="right",
            va="top")
    ax.text(0.03, 0.212, "80% 目标", fontsize=6.6, color=INK, va="bottom")
    ax.text(0.97, 0.212, "动物口径更贵", fontsize=6.6, color=ANIMAL,
            ha="right", va="bottom")
    ax.set_ylabel("倍数（票口径÷动物口径）", labelpad=2)
    top = ax.twiny()
    top.set_xlim(100, 0)
    top.set_xticks([80, 50, 20])
    top.set_xticklabels(["80%", "50%", "20%"], fontsize=7.2)
    top.set_xlabel("固定成本占比（1 − γ）", fontsize=8.0, labelpad=2)
    top.spines["top"].set_linewidth(0.6)
    top.tick_params(axis="x", width=0.6, length=2.4, pad=2)
    panel(ax, "c", dx=-0.19, dy=1.19)

    ax = fig.add_subplot(gs[1, 1])
    rows = d["by_target"]
    ratio_axes(ax, (0.2, 12.0))
    for key, col in (("t80", "#08519C"), ("t70", "#4292C6"), ("t60", "#9ECAE1")):
        yy = np.array([r[key] for r in rows])
        ax.plot(gx, yy, "-", color=col, lw=1.4 if key == "t80" else 1.0, zorder=3)
        ax.plot(gx, yy, "o", ms=1.5, color=col, zorder=4)
        c = d["cross_by_target"][key]["gamma"]
        ax.plot(c, 1.0, "o", ms=3.4, color=col, mec="white", mew=0.5, zorder=6)
        ax.plot([], [], "-", color=col, lw=1.2, label=key[1:] + "% 动物捕获")
    ax.legend(loc="upper right", frameon=False, handlelength=1.1,
              borderaxespad=0.25, labelspacing=0.25, fontsize=6.6)
    cs = [d["cross_by_target"][k]["gamma"] for k in ("t60", "t70", "t80")]
    ax.set_title(f"穿越点 γ = {min(cs):.2f}–{max(cs):.2f}", fontsize=7.0, pad=3,
                 color=INK)
    ax.set_ylabel("倍数（票口径÷动物口径）", labelpad=2)
    panel(ax, "d", dx=-0.19, dy=1.19)

    save(fig, "fig3_main.png")


def sankey_en(S) -> None:
    """Twin sankey on a 170 mm canvas so the left country column fits."""
    d = json.loads((RES / "p6_sankey.json").read_text(encoding="utf-8"))
    s = d["scope"]
    bands = {r["label"]: r for r in d["bands"]}
    W, H = 170.0, 96.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    S.panel(ax, d, 5.5, 82.5,
            "票口径：宽度＝拒收行数",
            f"{s['refused_rows']:,} 条拒收行 · {s['n_ports']} 个口岸",
            "rows",
            f">1000 档只占 {bands['>1000']['share_ticket']*100:.1f}% 的行",
            S.BLUE)
    S.panel(ax, d, 87.5, 164.5,
            "动物口径：宽度＝被拒动物数",
            f"{s['refused_animals_capped']:,.0f} 只（单行封顶 1 万）",
            "animals",
            f">1000 档占 {bands['>1000']['share_animal']*100:.1f}% 的动物",
            S.ORANGE)

    ax.plot([84.5, 84.5], [8.0, 76.0], color=S.LGREY, lw=0.6, ls=(0, (3, 3)),
            zorder=1)
    S.bold(ax, 84.5, 89.6, "同一批拒收，两种计量结构", fontsize=7.6,
           color=S.INK, ha="center", va="center")
    save(fig, "fig2_sankey.png")


def save(fig, name: str) -> None:
    for es in ("png", "pdf"):
        fig.savefig(FIG_EN / name.replace(".png", "." + es), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=600)
    plt.close(fig)
    print("wrote", name, "+ pdf")


def main() -> None:
    FIG_EN.mkdir(parents=True, exist_ok=True)
    import figures_core as F
    import sankey_fig as S
    def _ratio_axes(ax, ylim):  # noqa: ANN001
        ax.set_yscale("log")
        ax.axhspan(1.0, ylim[1], color=F.CASE, alpha=0.06, lw=0)
        ax.axhspan(ylim[0], 1.0, color=F.ANIMAL, alpha=0.07, lw=0)
        ax.axhline(1.0, color=F.GREY, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_ylim(*ylim)
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params(axis="y", which="minor", length=0)
        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("成本指数 γ（单行检查成本 ∝ $q^{γ}$）")


    S.CN = {k: k for k in S.CN}
    S.cn = lambda name: name
    S.BAND_CN = {"1–10": "1–10 items", "11–100": "11–100 items",
                 "101–1000": "101–1000 items", ">1000": ">1000 items"}

    mpl.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"

    leftover: list[str] = []
    orig = mpl.figure.Figure.savefig

    # the Chinese builder names the robustness panel "defence"; the English
    # edition writes it under a neutral name
    stem_en = {
        "fig1_institutional_gap": "fig1_shipment_vs_animals",
        "fig2_sankey": "fig2_source_structure",
        "fig2_dual_objective": "fig3_two_objectives",
        "fig4_defence": "fig4_robustness",
        "fig5_purpose": "fig5_purpose",
        "fig3_main": "fig6_frontier",
    }

    def savefig_en(self, fname, *a, **kw):  # noqa: ANN001
        f = Path(fname)
        if f.name == "fig2_dual_objective.png":
            fix_lists(self)
        leftover.extend(translate_fig(self))
        orig(self, FIG_EN / (stem_en.get(f.stem, f.stem) + f.suffix), *a, **kw)

    mpl.figure.Figure.savefig = savefig_en
    try:
        F.fig_gap()
        F.fig_lists()
        F.fig_defence()
        fig_purpose_en(F.CM, F.CASE, F.ANIMAL, F.INK, F.panel)
        sankey_en(S)
        fig3_en(F.CM, F.CASE, F.ANIMAL, F.INK, F.GREY, F.LGREY, F.panel,
                _ratio_axes)
    finally:
        mpl.figure.Figure.savefig = orig
    plt.close("all")

    if leftover:
        print("UNTRANSLATED TEXT ARTISTS:")
        for s in sorted(set(leftover)):
            print("   ", repr(s))
        sys.exit(1)
    print("english figures written to", FIG_EN)


if __name__ == "__main__":
    main()
