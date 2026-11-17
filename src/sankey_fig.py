"""Figure 2 - the same refusals, two units, two worlds (Sankey twin panel).

Left panel  link width = refused rows (2,486)
Right panel link width = refused animals, capped at 1e4 per row (212,670)

Columns: origin country (Top 8 + unknown origin + other) -> port (Top 6 +
other) -> size band (1-10 / 11-100 / 101-1000 / >1000).
The band column carries the message: the same refusals look like a
small-consignment world under one unit and a large-consignment world under the
other.

v2 replaces the v1 sketch (make_p6_sankey_sketch.py, kept for the record):
ports labelled by code + original English name, blue ramp on the left / orange
ramp on the right (blue = ticket, orange = animal throughout the paper),
link-slot layout rewritten (v1 mis-stacked the port -> band ribbons and let
them run past the canvas), label spreading for small nodes, caption line on
the country ordering.

Figure 2 of the paper: the same refusals under two measurement structures.
Input: results/p6_sankey.json (scripts/p6_sankey_agg.py)

Run: python make_p6_sankey_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
INK, GREY, LGREY = "#1A1A1A", "#8C8C8C", "#D9D9D9"


def bold(ax, x, y, s, **kw):  # noqa: ANN001
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, s, **kw)

PAPER = HERE.parent
RES = PAPER / "results"
FIG = PAPER / "figures"

BLUE = ["#C9E0EE", "#8CBBD6", "#3E86AE", "#0B4F76"]
ORANGE = ["#FBE3D3", "#F2BE9B", "#DE7B3C", "#B0450C"]
BAND_ORDER = ["1–10", "11–100", "101–1000", ">1000"]
BAND_CN = {"1–10": "1–10 件", "11–100": "11–100 件",
           "101–1000": "101–1000 件", ">1000": ">1000 件"}

CN = {
    "Taiwan (Province of China)": "中国台湾", "China": "中国", "Peru": "秘鲁",
    "Mexico": "墨西哥", "Unknown origin": "原产国不明", "Indonesia": "印度尼西亚",
    "Belgium": "比利时", "Turkey": "土耳其", "Thailand": "泰国",
    "Madagascar": "马达加斯加", "Germany": "德国", "Egypt": "埃及",
    "Togo": "多哥", "Ukraine": "乌克兰", "Central African Republic": "中非",
    "Congo": "刚果", "Vietnam": "越南", "Philippines": "菲律宾",
    "Australia": "澳大利亚", "Netherlands": "荷兰", "France": "法国",
    "Italy": "意大利", "Spain": "西班牙", "Japan": "日本", "Canada": "加拿大",
    "South Africa": "南非", "Brazil": "巴西", "Colombia": "哥伦比亚",
    "Ecuador": "厄瓜多尔", "Ghana": "加纳", "Nigeria": "尼日利亚",
    "Kenya": "肯尼亚", "Tanzania": "坦桑尼亚", "Malaysia": "马来西亚",
    "Singapore": "新加坡", "India": "印度", "Israel": "以色列", "Poland": "波兰",
}


def cn(name: str) -> str:
    return CN.get(name, name)


def port_label(code: str, name: str) -> str:
    return f"{code} · {name.replace('. FL', '')}"


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha, z=2):
    """Cubic ribbon between two vertical spans."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color,
                           edgecolor="none", alpha=alpha, zorder=z))


def stack(values, names, y_top, y_bot, gap):
    """Return {name: (y_bottom, y_top)} filling the column top-down."""
    total = sum(values.values())
    usable = (y_top - y_bot) - gap * (len(names) - 1)
    out, y = {}, y_top
    for n in names:
        h = usable * values[n] / total
        out[n] = (y - h, y)
        y -= h + gap
    return out


def spread(centers, min_gap):
    """Push overlapping label centres apart, keeping the order."""
    ys = list(centers)
    for _ in range(300):
        moved = False
        for i in range(1, len(ys)):
            d = ys[i - 1] - ys[i]
            if d < min_gap:
                sh = (min_gap - d) / 2
                ys[i - 1] += sh
                ys[i] -= sh
                moved = True
        if not moved:
            break
    return ys


def panel(ax, d, x_left, x_right, title, subt, measure, callout, palette):
    flows = d["flows_all"]
    val = (lambda r: r["refu_rows"]) if measure == "rows" else (lambda r: r["cap_animals"])

    c_tot, p_tot, b_tot, p_code = {}, {}, {}, {}
    for r in flows:
        c, p, b = r["org_name"], r["port_name"], r["band_label"]
        v = val(r)
        c_tot[c] = c_tot.get(c, 0) + v
        p_tot[p] = p_tot.get(p, 0) + v
        b_tot[b] = b_tot.get(b, 0) + v
        p_code[p] = r["port"]

    c_rank = sorted(c_tot, key=c_tot.get, reverse=True)
    keep_c = c_rank[:8]
    if "Unknown origin" not in keep_c and "Unknown origin" in c_tot:
        keep_c = c_rank[:7] + ["Unknown origin"]
        keep_c.sort(key=c_tot.get, reverse=True)
    c_names = keep_c + ["其他"]
    c_vals = {c: c_tot[c] for c in keep_c}
    c_vals["其他"] = sum(c_tot[c] for c in c_tot if c not in keep_c)

    p_rank = sorted(p_tot, key=p_tot.get, reverse=True)
    keep_p = p_rank[:6]
    p_names = keep_p + ["其他"]
    p_vals = {p: p_tot[p] for p in keep_p}
    p_vals["其他"] = sum(p_tot[p] for p in p_tot if p not in keep_p)

    b_names = BAND_ORDER
    b_vals = {b: b_tot.get(b, 0) for b in b_names}

    y_top, y_bot = 74.0, 14.0
    cpos = stack(c_vals, c_names, y_top, y_bot, 1.3)
    ppos = stack(p_vals, p_names, y_top, y_bot, 1.6)
    bpos = stack(b_vals, b_names, y_top, y_bot, 1.8)

    xc, xp, xb = x_left + 10.0, x_left + 36.0, x_left + 60.0
    nw = 1.4

    # country -> port: slot offsets advance independently on each side
    c_off = {c: cpos[c][1] for c in c_names}
    p_off = {p: ppos[p][1] for p in p_names}
    for c in c_names:
        for p in p_names:
            v = sum(val(r) for r in flows if r["org_name"] == c and r["port_name"] == p)
            if not v:
                continue
            hc = (cpos[c][1] - cpos[c][0]) * v / c_vals[c]
            hp = (ppos[p][1] - ppos[p][0]) * v / p_vals[p]
            y0a, y0b = c_off[c], c_off[c] - hc
            y1a, y1b = p_off[p], p_off[p] - hp
            c_off[c], p_off[p] = y0b, y1b
            by_b = {b: sum(val(r) for r in flows
                           if r["org_name"] == c and r["port_name"] == p
                           and r["band_label"] == b) for b in BAND_ORDER}
            dom = max(by_b, key=by_b.get)
            ribbon(ax, xc + nw, xp, y0a, y0b, y1a, y1b,
                   palette[BAND_ORDER.index(dom)], 0.55)

    # port -> band
    p_off = {p: ppos[p][1] for p in p_names}
    b_off = {b: bpos[b][1] for b in b_names}
    for p in p_names:
        for b in BAND_ORDER:
            v = sum(val(r) for r in flows if r["port_name"] == p and r["band_label"] == b)
            if not v:
                continue
            hp = (ppos[p][1] - ppos[p][0]) * v / p_vals[p]
            hb = (bpos[b][1] - bpos[b][0]) * v / b_vals[b]
            y0a, y0b = p_off[p], p_off[p] - hp
            y1a, y1b = b_off[b], b_off[b] - hb
            p_off[p], b_off[b] = y0b, y1b
            ribbon(ax, xp + nw, xb, y0a, y0b, y1a, y1b,
                   palette[BAND_ORDER.index(b)], 0.92, z=3)

    # nodes
    for c in c_names:
        y0, y1 = cpos[c]
        ax.add_patch(Rectangle((xc, y0), nw, y1 - y0, facecolor="#404040",
                               edgecolor="none", zorder=5))
    for p in p_names:
        y0, y1 = ppos[p]
        ax.add_patch(Rectangle((xp, y0), nw, y1 - y0, facecolor="#404040",
                               edgecolor="none", zorder=5))
    for i, b in enumerate(b_names):
        y0, y1 = bpos[b]
        ax.add_patch(Rectangle((xb, y0), nw, y1 - y0, facecolor=palette[i],
                               edgecolor="#8C8C8C", lw=0.25, zorder=5))

    # labels, spread so that small nodes stay readable
    for c, y in zip(c_names, spread([(cpos[c][0] + cpos[c][1]) / 2
                                     for c in c_names], 2.5)):
        ax.text(xc - 1.6, y, cn(c), fontsize=5.2, color=INK, ha="right",
                va="center", zorder=6)
    for p, y in zip(p_names, spread([(ppos[p][0] + ppos[p][1]) / 2
                                     for p in p_names], 2.5)):
        txt = "其他" if p == "其他" else port_label(p_code[p], p)
        ax.text(xp + nw + 1.1, y, txt, fontsize=4.8, color=INK, va="center",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.80))
    for b, y in zip(b_names, spread([(bpos[b][0] + bpos[b][1]) / 2
                                     for b in b_names], 4.0)):
        ax.text(xb + nw + 1.2, y, BAND_CN[b], fontsize=5.2, color=INK,
                va="center", zorder=6)

    cx = (x_left + x_right) / 2
    bold(ax, cx, 82.0, title, fontsize=7.8, color=INK, ha="center", va="center")
    ax.text(cx, 78.4, subt, fontsize=5.6, color=GREY, ha="center", va="center")
    bold(ax, cx, 10.0, callout, fontsize=6.0, color=INK, ha="center",
         va="center")


def main() -> None:
    d = json.loads((RES / "p6_sankey.json").read_text(encoding="utf-8"))
    s = d["scope"]
    bands = {r["label"]: r for r in d["bands"]}

    try:
        mpl.font_manager.fontManager.addfont("C:/Windows/Fonts/msyhbd.ttc")
    except Exception:
        pass
    mpl.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.size": 6.0, "pdf.fonttype": 42, "savefig.dpi": 600,
    })
    W, H = 163.0, 96.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    panel(ax, d, 2.0, 79.0,
          "票口径：宽度＝拒收行数",
          f"{s['refused_rows']:,} 条拒收行 · {s['n_ports']} 个口岸",
          "rows",
          f">1000 档只占 {bands['>1000']['share_ticket']*100:.1f}% 的行",
          BLUE)
    panel(ax, d, 84.0, 161.0,
          "动物口径：宽度＝被拒动物数",
          f"{s['refused_animals_capped']:,.0f} 只（单行封顶 1 万）",
          "animals",
          f">1000 档占 {bands['>1000']['share_animal']*100:.1f}% 的动物",
          ORANGE)

    ax.plot([81.5, 81.5], [8.0, 76.0], color=LGREY, lw=0.6, ls=(0, (3, 3)),
            zorder=1)
    bold(ax, 81.5, 89.6, "同一批拒收，两种计量结构", fontsize=7.6, color=INK,
         ha="center", va="center")

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_sankey.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    print("wrote", FIG / "fig2_sankey.png")


if __name__ == "__main__":
    main()
