# Reproduction map

Every number in the paper's tables and every panel in its figures is produced
by the scripts below and stored in `results/`. No value is typed by hand.

Run everything with `python src/run_all.py`; run one stage with
`python src/<stage>.py`.

## Tables

| Table | Contents | Producer | Keys in `results/` |
|---|---|---|---|
| 2 | Sample construction (all rows → whole animals → test period) | `descriptives.py` | `p6_descriptives.json` · `T_all`, `T_whole`, `T_whole_test` |
| 3 | Notation | — | definitions only |
| 4 | Main results at a 10% budget | `analysis.py`, `revisions.py` | `p6_main.json` · `bootstrap.point`, `bootstrap.ci95`; `p6_revisions.json` · `X1_quantity_only` |
| 5 | Budget for 80% animal capture by cost regime | `upgrade.py` | `p6_shadow_cost.json` · `gamma0.0`, `gamma0.5`, `gamma1.0`, `log`, `ratio_80anim` |
| S1 | Rolling test years 2013–2022 | `analysis.py rolling` | `p6_rolling.json` · per-year `rows`, `refu`, `auc`, `at10` |
| S2 | Robustness checks (caps, alternative rankers, recheck, personal imports) | `analysis.py`, `metric_robustness.py` | `p6_main.json` · `alt_score`, `shipment`; `p5b_metric_robustness.json` · `raw`, `cap_1000`; `p6_purpose_p.json` |
| S3 | Deleting the largest refused rows | `upgrade.py delete_top_ci` | `p6_delete_top_ci.json` · `variants.drop{0,1,10,100}.point` and `.ci95` |
| S4 | Budget sensitivity 5/10/15/20% | `analysis.py sweep` | `p6_main.json` · `sweep.curve` |
| S5 | Six shipments traced to raw rows | `case_audit.py` | `p6_case_audit.json` · `cases` |

## Figures

| Figure | Contents | Producer | Keys in `results/` |
|---|---|---|---|
| 1 | Size bands, concentration curve, refusal-rate gradient | `figures.py` (panel data: `figdata.py`) | `p6_figdata.json` · `T_test_concentration` and band tables |
| 2 | Origin → port → size Sankey, two panels | `sankey.py`, `sankey_fig.py` | `p6_sankey.json` · `bands`, `countries`, `ports`, `flows_top` |
| 3 | Budget sweep, four-way decomposition, list composition, year-by-year difference | `analysis.py`, `figures.py` | `p6_main.json` · `sweep`, `lambda`, `composition`; `p6_rolling.json` |
| 4 | Robustness across caps, rankers, recheck, personal imports | `analysis.py`, `metric_robustness.py`, `figures.py` | `p6_main.json` · `alt_score`, `shipment`; `p5b_metric_robustness.json`; `p6_purpose_p.json` |
| 5 | Commercial versus personal imports | `figures.py` | `p6_purpose_p.json` (produced inside `analysis.py purpose_p`) |
| 6 | Shadow cost and the objective–cost frontier | `upgrade.py`, `analysis.py`, `figures.py` | `p6_shadow_cost.json` · cost regimes; `p6_main.json` · `lambda` |

## Conventions

- **Budget** is always a share of declaration *rows*, not of shipments. The
  shipment-level recheck aggregates rows by `control_number` first.
- **Capping**: the per-row quantity cap is 10,000, estimated in the training
  period and fixed before the test period. `analysis.py` applies it to the
  animal metric; `upgrade.py` recomputes the whole pipeline for `raw` and
  `1,000`.
- **Bootstrap**: 500 replications clustered by species, fixed seed, applied to
  every reported capture rate and gap.
- **Rolling years**: for each test year *y*, the model is fitted on all years
  before *y − 1*, the year *y − 1* is used for early stopping, and *y* is
  scored.
- **Random seeds**: `SEED = 7` in `src/data_layer.py`, passed to LightGBM.
