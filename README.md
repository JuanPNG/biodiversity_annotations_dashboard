# Biodiversity—Genome Annotations Dashboard

Interactive Plotly Dash app for exploring biodiversity genomics across taxonomy, biogeography, and (soon) climate.

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
│   ├── styles.css              # Layout/theme (filters grid, collapsible, buttons)
│   └── custom.js               # (optional) currently unused for links
│
├── pages/
│   ├── home.py                 # Landing page
│   ├── data_browser.py         # AG Grid page (server-side paging/filters)
│   └── genome_annotations.py   # Stacked bar + drill-down for gene biotypes
│
├── layouts/
│   └── navbar.py               # Top navbar + global filters (taxonomy, biogeo)
│
├── callbacks/
│   ├── global_filters.py       # Populate/cascade filters, reset button, global store
│   ├── data_browser_callbacks.py
│   └── genome_annotations_callbacks.py
│
├── utils/
│   ├── config.py               # Central config (paths, column names, presets)
│   ├── parquet_io.py           # Arrow dataset scans, filters, aggregations
│   └── data_tools.py           # Helper queries (distincts, cascades)
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
### Global filters (navbar)

* Taxonomy multi-select for each rank (kingdom→species + tax_id)
  * Cascading options: lower ranks show only values valid under selections above 
  * OR within a rank, AND across ranks 
  * Reset taxonomy button clears all rank selections
* Biogeography: choose level(s) then value(s); resolves to an accession allow-list
* Selections are stored centrally in dcc.Store(id="global-filters") and respected by all pages

### Data Browser (AG Grid)

* Reads only needed columns and rows with PyArrow predicate pushdown + column pruning
* Server-side paging (page & size controls)
* Column presets & selector in a collapsible panel (hidden by default) with a live “(N selected)” badge
* Clickable links for URL columns (via built-in markdown renderer)
* Status line: page, size, rows returned, filtered total, and active ranks

### Genome Annotations

* 100% stacked horizontal bar chart of gene biotype composition per selected taxon rank
* Uses *_count columns and computes percentages per group; if total_gene_biotypes exists, we normalize by it (stacks reflect true totals; may sum <100% if not all genes are in tracked biotypes)
* Drill down / up:
  * Click a bar segment to drill to the next rank 
  * “Up one level” moves back 
  * Robust click handling via Plotly customdata
* Honors global taxonomy & biogeo filters
* Default grouping: Kingdom

### How it works (run time flow)
1. **Global filters** update `global-filters` store:
```
{
  "taxonomy_map": {"family": ["Rosaceae"], "genus": ["Rosa"]},
  "bio_levels": ["biome"],
  "bio_values": ["Temperate Broadleaf..."]
}
```
2. Scanners stream batches:

* **Grid**: fetch paged slice with selected columns
* **Genome annotations**: aggregate `*_count` per group → compute % (optionally using total_gene_biotypes). 