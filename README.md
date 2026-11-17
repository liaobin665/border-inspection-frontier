# Border inspection frontier

**Shipment-based versus animal-based ranking of border wildlife inspections.**

This repository reproduces the analysis behind a study of a single question: at
a fixed risk score and a fixed inspection budget, does the *unit* in which
inspection performance is measured — one shipment or one animal — change which
consignments are selected, and what does that switch cost?

It contains the full pipeline (data layer, model, all robustness checks, all
figures) but no raw records. See [docs/DATA.md](docs/DATA.md) for how to obtain
the public LEMIS extract and [docs/REPRODUCE.md](docs/REPRODUCE.md) for the
script-to-table/figure map.

中文说明：[README.zh-CN.md](README.zh-CN.md)

---

## 1. What the analysis does

1. **Scope.** U.S. LEMIS import declarations, commercial purpose, 2010–2022,
   narrowed to whole animals (`LIV`, `BOD`, `DEA`, `TRO`, `SKE`, `FIG`).
2. **One ranker.** A LightGBM classifier estimates a historical refusal
   propensity for every declaration row, using only information from earlier
   years. Test period 2019–2022, AUC 0.898, AP 0.387.
3. **Two objectives, same model.** The shipment ranking orders rows by `p`; the
   animal ranking orders them by `p × q`. Both are evaluated at the same
   budget, so the only thing that changes is the objective.
4. **Frontier.** A weighted score `p × q^λ` traces the trade-off between the
   two outputs; a cost index `γ` (cost ∝ `q^γ`) translates the frontier into
   inspection capacity.
5. **Robustness.** Ten rolling test years, 500 species-clustered bootstrap
   replications, three capping regimes, three alternative rankers, a
   shipment-level recheck and a personal-import boundary case.

The headline numbers: at a 10% budget the two lists overlap at a Jaccard of
0.200; the animal ranking captures 41.5 percentage points more refused animals
and 24.0 points fewer refused shipments; under per-row costing the shipment
ranking needs 6.7 times the budget to reach the same animal capture, and that
advantage reverses once size costs pass roughly half of a typical inspection.

## 2. What is in the box

| Path | Contents |
|---|---|
| `src/` | Pipeline, one module per stage (see below) |
| `results/` | Aggregate results (JSON) — the only inputs the figures need |
| `figures/` | Six figures, PDF and PNG, as printed |
| `docs/` | Data acquisition and reproduction notes |

| Module | Stage |
|---|---|
| `data_layer.py` | LEMIS query, scope, features, historical rates, train/valid/test split |
| `whole_animals.py` | Whole-animal scope and concentration statistics |
| `metric_robustness.py` | Capping and metric variants |
| `analysis.py` | Main results, bootstrap, budget sweep, λ frontier, composition, alternative rankers, shipment recheck, personal imports |
| `upgrade.py` | Delete-largest-row bootstrap and shadow-cost scenarios |
| `revisions.py` | Quantity-only baseline, cost scenarios, shipment-mix regression |
| `descriptives.py` | Sample construction by purpose and scope |
| `sankey.py` | Origin–port–size flows behind Figure 2 |
| `figdata.py` | Panel data for Figure 1 |
| `split_probe.py` | Train/validation protocol probe and ROC data |
| `case_audit.py` | Six shipments traced back to raw rows |
| `figures_core.py`, `sankey_fig.py` | Panel drawing (language-neutral) |
| `figures.py` | English lettering and file naming (Figure 1–6) |
| `run_all.py`, `build_db.py` | Orchestration and database loader |

Not included: raw LEMIS records, the DuckDB file, per-row tables, fitted model
binaries.

## 3. Quick start

```bash
pip install -r requirements.txt
python src/build_db.py path/to/lemis.csv       # writes data/lemis.duckdb
python src/run_all.py                          # results/ and figures/
python src/run_all.py --figures                # figures only
```

The full run trains one LightGBM model several times (main fit plus ten rolling
years) and re-runs a 500-replication bootstrap; on a laptop this takes tens of
minutes. Figures can be rebuilt from `results/` alone.

Set `LEMIS_DB=/path/to/lemis.duckdb` to keep the database outside the
repository.

## 4. Reproducibility notes

- Seeds are fixed (`SEED = 7` in `src/data_layer.py`).
- Historical rates use only years strictly before a row's own year and are
  shrunk towards a training-period prior, so no feature sees the future.
- Every number in `results/` is an aggregate; no record-level data are stored.
- Figures are drawn from `results/` only, so a figure can be rebuilt without
  touching the database.

## 5. License

MIT — see [LICENSE](LICENSE). The LEMIS data themselves are subject to the
terms of their release and are not covered by this license.
