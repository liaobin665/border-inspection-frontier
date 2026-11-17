# 边境检查前沿（Border Inspection Frontier）

**边境野生动物检查：按票排序与按动物排序的比较。**

本仓库复现一项研究：在同一个风险分数、同一笔检查预算下，把绩效计量单位从"一票"换成"一只"，会不会改变被选中的货物清单，代价有多大。

仓库包含完整流水线（数据层、模型、全部稳健性检验、全部出图），但不含原始记录。数据获取见 [docs/DATA.md](docs/DATA.md)，"脚本 → 表/图"对照见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

English: [README.md](README.md)

---

## 1. 分析做了什么

1. **口径**：美国 LEMIS 进口申报，商业用途，2010–2022，收窄到整只动物（`LIV`/`BOD`/`DEA`/`TRO`/`SKE`/`FIG`）。
2. **一个排序器**：LightGBM 估计每一行的"历史拒收倾向"，只使用早于该行年份的信息。测试期 2019–2022，AUC 0.898，AP 0.387。
3. **两个目标、同一个模型**：票口径按 `p` 排序，动物口径按 `p × q` 排序；两者在同一预算下评估，唯一变化是目标函数。
4. **前沿**：加权分数 `p × q^λ` 连接两个端点；成本指数 `γ`（单行成本 ∝ `q^γ`）把前沿换算成检查能力。
5. **稳健性**：10 个滚动测试年、500 次物种聚类 bootstrap、三种截断、三种替代排序器、整票复核、个人进口对照。

主要读数：10% 预算下两份名单的 Jaccard 重合为 0.200；动物口径多捕获 41.5 个百分点的被拒动物、少捕获 24.0 个百分点的被拒票；按行计费时，票口径要达到同样的动物捕获需 6.7 倍预算，而当规模成本占单次检查约一半以上时这一优势反转。

## 2. 目录

| 路径 | 内容 |
|---|---|
| `src/` | 流水线，一个阶段一个模块 |
| `results/` | 聚合结果（JSON），出图的唯一输入 |
| `figures/` | 六张图，PDF 与 PNG，按印刷尺寸 |
| `docs/` | 数据获取与复现说明 |

模块职责见英文 README 的同名表格（`data_layer` → `figures` → `run_all`）。

不含：原始 LEMIS 记录、DuckDB 文件、逐行分析表、模型二进制。

## 3. 快速开始

```bash
pip install -r requirements.txt
python src/build_db.py path/to/lemis.csv       # 生成 data/lemis.duckdb
python src/run_all.py                          # 重算 results/ 与 figures/
python src/run_all.py --figures                # 只重出图（需 results/）
```

完整跑一遍要多次训练 LightGBM（主模型 + 10 个滚动年）并做 500 次重抽，笔记本上需数十分钟；只出图则几分钟。

## 4. 复现约定

- 种子固定（`src/data_layer.py` 中 `SEED = 7`）。
- 历史拒收率只用严格早于该行年份的数据，并向训练期先验收缩，特征不看未来。
- `results/` 里全部是聚合量，没有记录级数据。
- 图只读 `results/`，可以脱离数据库重出。

## 5. 许可

MIT，见 [LICENSE](LICENSE)。LEMIS 数据本身适用其发布条款，不在本许可范围内。
