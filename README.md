# Biodiversity—Genome Annotations Dashboard

Interactive Plotly Dash app for exploring biodiversity genomics across taxonomy, biogeography, and climate.

Optimized for large Parquet datasets using PyArrow with predicate pushdown, column pruning, and server-side paging.

## Quick start 

```bash 
# Python 3.10+
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Run dev server
python app.py
# Dash will start at http://0.0.0.0:8050/
```
## Project structure

```
.
├── app.py                      # Creates Dash app (use_pages=True) and runs it
├── requirements.txt
├── README.md
│
├── assets/                     # Auto-served by Dash
│   └── styles.css              # Layout/theme (filters grid, collapsible, buttons)
│
├── pages/
│   ├── home.py                 # Landing page
│   ├── data_browser.py         # AG Grid page (server-side paging/filters)
│   ├── genome_annotations.py   # Stacked bar + drill-down for gene biotypes
│   └── biotype_environment.py  # Scatterplots for gene biotypes vs environmental variables. 
│
├── layouts/
│   └── navbar.py               # Top navbar + global filters (taxonomy, biogeo)
│
├── callbacks/
│   ├── global_filters.py       # Populate/cascade filters, reset button, global store
│   ├── data_browser_callbacks.py
│   ├── genome_annotations_callbacks.py
│   ├── home_kpis.py
│   └── ui_badges.py
│
├── utils/
│   ├── config.py               # Central config (paths, column names, presets)
│   ├── parquet_io.py           # Arrow dataset scans, filters, aggregations
│   ├── data_tools.py           # Helper queries (distincts, cascades)
│   └── types.py                # Aliases for shared data shapes.
│ 
├── tests/
│   └──  test_global_filters_helpers    # Unit testing for global filters. 
│
└── data/
    └── processed/              # **Parquet** files (app reads from here)
        ├── dashboard_main.parquet
        ├── biogeo_long.parquet
        └── gbif_occurrences.parquet
```

## Data model (columns)

**Main table** (`dashboard_main.parquet`)
* Taxonomy ranks: kingdom, phylum, class, order, family, genus, species, tax_id
* Gene biotypes: *_count for each biotype (e.g., protein_coding_count, lncRNA_count, …) and total_gene_biotypes
* Climate stats (examples): clim_bio1_mean/max/min, clim_bio7_*, clim_bio12_*, clim_bio15_*
* Distribution: range_km2, mean_elevation, min_elevation, max_elevation, median_elevation
* Metadata/URLs: biodiversity_portal, gtf_file, ensembl_browser, gbif

**Biogeography long** (`biogeo_long.parquet`)
* accession, level, value (where level ∈ realm/biome/ecoregion; value is the category)

**GBIF occurrences** (`gbif_occurrences.parquet`)
* Occurrence points (used later for maps)

## Configuration
Set the key settings used across the app in `utils/config.py`

## Features
### Global filters

* Backed by `dcc.Store(id="global-filters")` and respected by all pages.
* Taxonomy multi-select for each rank (kingdom→species + tax_id)
  * Cascading options: lower ranks show only values valid under selections above 
  * OR within a rank, AND across ranks 
  * Reset taxonomy button clears all rank selections
* Biogeography: choose level(s) then value(s); resolves to an accession allow-list + **range_km2** numeric slider.
* Climate categorical lables (placeholder in UI) + numeric sliders
  * `clim_bio1_mean` (temp)
  * `clim_bio12_mean` (precip)
* **Gene biotype %** filter using `*_percentage`columns; full-span (0–100) is treated as “no filter”.

**Data-driven slider spans:** min/max come from the dataset (no hard-coding).
**Full-span rule:** exact full span means “no filter” → key omitted from the store.

### Pages 

#### Home

* Welcoming and short introduction to the dashboard.
* Data KPIs: counts per taxonomic rank/biogeo, total genes, top biotypes. Robust to empty results.
* Honors Global Filters.
* Exploratory cards.
* Links for contact and feedback.

#### Data Browser (AG Grid)

* Reads only needed columns and rows with PyArrow predicate pushdown + column pruning
* Server-side paging (page & size controls)
* Column presets & selector in a collapsible panel (hidden by default) with a live “(N selected)” badge
* Clickable links for URL columns (via built-in markdown renderer)
* Status line: page, size, rows returned, filtered total, and active ranks

#### Genome Annotations

* 100% stacked horizontal bar chart of gene biotype composition per selected taxon rank
* Uses the precomputed `*_percentage` column or in absence uses `*_count` columns to percentages per group; if total_gene_biotypes exists, we normalise by it (stacks reflect true totals; may sum <100% if not all genes are in tracked biotypes)
* Drill down / up / reset buttons.
* Click a bar segment to drill to the next rank
* Honors global filters
* Default grouping: Kingdom

#### Biotype vs Environment

* two scatter plots (climate vs biotype, distribution vs biotype)
* Optional climate variable and distribution variable on X axis. 
* Optional metrics: Percentage, Raw Count, and Per-1k genes (default). 
* Optional point size by total genes
* Regression overlay (OLS) (Visual only - Statistical analyses pending)
* Allows log x/y axes.
* Tooltip: species name, accession, gene biotype aggregation metric, and environmental metric. 

### Performance & I/O

* PyArrow Datasets for pushdown + column pruning. 
* Server-side pagination in the Data Browser. 
* Micro-caching (process-local) only where safe:
* list_taxonomy_options() (no args) → @lru_cache(maxsize=1)
* Column min/max extents via parquet_io.get_column_min_max() use an internal cached scanner 
* Avoid caching callables that accept unhashable args (e.g., dicts).

### Typing & structure

* Python 3.10+ typing: built-in generics (dict, list, tuple), T | None.
* Central type aliases in utils/types.py:
  * `GlobalFilters`, `TaxonomyMap`, `ClimateRanges`, `BiogeoRanges`, `BiotypePctFilter`.
* Helpers use docstrings and typed signatures; callbacks stay thin.
* Explicit public APIs via __all__ in utils/types.py, utils/data_tools.py, utils/parquet_io.py.

### Global filters store contract

Keys are present only when non-empty (omit full-span sliders):

```python
{
  "taxonomy_map": { "rank": ["values"], ... },
  "climate": ["labels"],
  "bio_levels": ["levels"],
  "bio_values": ["values"],
  "climate_ranges": { "clim_bio1_mean": [lo, hi], "clim_bio12_mean": [lo, hi] },
  "biogeo_ranges": { "range_km2": [lo, hi] },
  "biotype_pct": { "biotype": "<base>", "min": float, "max": float }
}

```
Implementation details:

* `gf_is_full_span(lo, hi, span_lo, span_hi)`
* `gf_build_climate_ranges(...)` and` gf_build_biogeo_ranges(...)` unpack slider values and omit full-span.
* `gf_build_biotype_pct(biotype, [lo, hi])` returns a dict only if narrowed; otherwise None.

### How sliders get their ranges

In `layouts/navbar.py`, climate and biogeography sliders are initialized from data extents:

* `utils.parquet_io.get_column_min_max([col])` → returns `{col: (min, max)}` 
* Sliders’ `min`, `max`, and initial `value=[min,max]` are set at import time 
* Resets restore to these extents, not to hard-coded constants

### How to add a new numeric slider (cheat sheet)

1. Navbar: add a new `_slider_from_col(...)` entry with col name and a fallback span.
2. `global_filters.sync_global_store`: add the new `Input(..., "value")` and `State(..., "min"/"max")`; then extend `gf_build_*_ranges` or create a new helper.
3. Store contract: ensure the helper omits full-span and returns `{col: (lo, hi)}`.
4. Downstream pages: reference the new range from `global-filters.data` when constructing predicates.

## Testing

Fast tests that cover pure helpers (no Parquet required):

* `tests/test_global_filters_helpers.py` (local):
  * `gf_is_full_span` strict behaviour 
  * climate/biogeo range builders (full-span omitted, narrowed included)
  * biotype% full-span omission 
  * global store packing (only non-empty keys; climate_labels persisted under "climate")

Run: `pytest -q`

## Coding standards

* Python ≥ 3.10; works on 3.12.
* Use aliases (ClimateRanges, BiogeoRanges, GlobalFilters) for shared shapes.
* Callbacks tiny, pure logic in utils/data_tools.py.
* I/O centralized in utils/parquet_io.py.
* No new heavy deps; avoid background/async behavior.
* IDs and store contract are stable—don’t change without discussion.

## Next steps

Short-term (non-breaking):

1. Climate categorical labels
   * Integrate external categorization (you’ll prepare this externally).
   * Populate filter-climate options and write selected labels into store key "climate".

2. Interactive Maps
   * Add a page with GBIF occurrences (gbif_occurrences.parquet) and spatial vector layers for realms/biomes/ecoregions. 
   * Likely stack: dash-leaflet or pydeck (lightweight) with server-side query of points. 
   * Tooltips: species, accession, biogeo tags, gene biotypes; respect global filters.

3. Micro-caching (later)
   * Flask-Caching for schema/domains/extents if needed for prod; keep process-local LRU for now.

Nice-to-have:
* CI (pytest + lint) and a “smoke run” workflow.
* Tiny perf polish: cache biotype pct column discovery.

## Known limitations
* Categorical climate labels are placeholders until external source is integrated.
* No map page yet (GBIF points and vector layers to be added).
* Process-local caches reset on app restart (fine for dev/prototype).