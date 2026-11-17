# Data

The repository does **not** distribute LEMIS records. It expects a DuckDB
database at `data/lemis.duckdb` with a table named `lemis`, built from the
public U.S. LEMIS release.

## Where the records come from

LEMIS (Law Enforcement Management Information System, U.S. Fish and Wildlife
Service) records every declaration of fish and wildlife imported into or
exported from the United States. The records are released under the Freedom of
Information Act:

- public description of the system and of the declaration form:
  <https://www.fws.gov/story/2025-08/service-declaration-importation-or-exportation-fish-and-wildlife>
- Office of Law Enforcement (FOIA requests):
  <https://www.fws.gov/program/office-of-law-enforcement>
- a documented, analysis-ready export of the same records:
  Eskew et al. (2020), *Scientific Data* 7:22,
  <https://doi.org/10.1038/s41597-020-0354-5>

Download or request the export, then load it:

```bash
python src/build_db.py path/to/lemis.csv        # writes data/lemis.duckdb
```

The loader keeps every column as text; all casting happens in the analysis
queries, so the exact CSV column order does not matter.

## Columns used

| Column | Used for |
|---|---|
| `import_export` | scope: `I` = import |
| `unit` | scope: `NO` = items |
| `purpose` | `T` = commercial, `P` = personal |
| `sYear`, `shipment_date` | year, month, weekday |
| `quantity` | declared items per row |
| `species`, `corrected` | species (the corrected name is used when present) |
| `country_origin`, `port` | origin and port of entry |
| `description` | item category (`LIV`, `BOD`, `DEA`, `TRO`, `SKE`, `FIG`, …) |
| `source` | wild / captive-bred / unknown |
| `action` | `R` = refused |
| `disposition` | `S` = seized |
| `control_number` | shipment id (present for 2019–2020 only) |
| `group_` | taxonomic group |

Rows whose species field reads `REMAINS UNKNOWN` (or contains `UNKNOWN`) are
dropped; see `src/data_layer.py::pull` for the exact filter.

## Optional: code tables and species status

`src/sankey.py` reads `data/codes.csv` with columns `field`, `code`,
`description` to label ports and countries. If the file is absent the script
falls back to the raw codes and still runs.

`src/revisions.py` reads `data/iucn_status.csv` with columns `species`,
`status` (any Red List category: `lc`, `nt`, `vu`, `en`, `cr`) to measure how
far a threat weighting could reach. If the file is absent the coverage is
reported as zero; the published 8.9% figure was computed with the cache filled
from the public IUCN Red List API.

## What is not shipped

Raw LEMIS records, the DuckDB file, per-row analysis tables, fitted models and
any intermediate arrays. Everything in `results/` is an aggregate: counts,
rates, confidence intervals and model scores, with no record-level data.
