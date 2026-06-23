"""
Central dashboard configuration.

This module describes the processed data schema used by the Dash app:
- where processed Parquet files live,
- which columns represent accessions, taxonomy, climate, biogeography, URLs,
- which columns are grouped into Data Browser presets,
- how gene biotype count/percentage columns are named.

Most modules import these constants instead of hard-coding column names.
If the ETL output schema changes, update this file first and then check the
affected readers in utils/parquet_io.py and page callbacks.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# Runtime data location. The app expects ETL outputs to be written here.
DATA_DIR = Path(os.getenv("DASHBOARD_DATA_DIR", "data/processed"))

# Processed Parquet files produced by etl_parquet/run_etl_parquet.py.
DASHBOARD_MAIN_FN = "dashboard_main.parquet"
BIOGEO_LONG_FN = "biogeo_long.parquet"
GBIF_OCCURRENCES_FN = "gbif_occurrences.parquet"  # not used yet in filters

# ---------------------------------------------------------------------------
# Processed table column contract
# ---------------------------------------------------------------------------

# Column names in dashboard_main.parquet.
# Keep these aligned with etl_parquet/prep_main_parquet.py.
ACCESSION_COL_MAIN = "accession"         # in dashboard_main
TAXONOMY_COL = None
TAXONOMY_RANK_COLUMNS = [
    "kingdom", "phylum", "class", "order", "family", "genus", "species"
]

# Climate label (TODO: Kopen classification)
CLIMATE_LABEL_COL = "clim_bio1_mean" # Provisional

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

# Used by Data Browser headers/dropdowns and other UI helpers. Keys must be
# actual dashboard_main.parquet column names.
COLUMN_LABELS: dict[str, str] = {
    # Climate variables
    "clim_bio1_mean": "Mean Annual Mean Temperature (°C)",
    "clim_bio1_max":  "Max Annual Mean Temperature (°C)",
    "clim_bio1_min":  "Min Annual Mean Temperature (°C)",
    "clim_bio12_mean": "Mean Annual Precipitation (mm)",
    "clim_bio12_max": "Max Annual Precipitation (mm)",
    "clim_bio12_min": "Min Annual Precipitation (mm)",
    "clim_bio7_mean":  "Mean Temperature Annual Range (°C)",
    "clim_bio7_max":  "Max Temperature Annual Range (°C)",
    "clim_bio7_min":  "Min Temperature Annual Range (°C)",
    "clim_bio15_mean": "Mean Precipitation Seasonality (CV)",
    "clim_bio15_max": "Max Precipitation Seasonality (CV)",
    "clim_bio15_min": "Min Precipitation Seasonality (CV)",

    # Distribution variables
    "range_km2": "Distribution Range Size (km², EOO)",
    "mean_elevation": "Mean Elevation (m)",
    "min_elevation": "Min Elevation (m)",
    "max_elevation": "Max Elevation (m)",
    "median_elevation": "Median Elevation (m)",
}


# biogeo_long.parquet columns used by global biogeography filters and KPIs.
# Column names in dashboard_main.parquet.
# Keep these aligned with etl_parquet/prep_main_parquet.py.
ACCESSION_COL_BIOGEO = "accession"       # in biogeo_long
BIOGEO_LEVEL_COL = "level"
BIOGEO_VALUE_COL = "value"
BIOGEO_LEVELS_TO_EXPOSE = ["realm", "biome", "ecoregion"]

# ---------------------------------------------------------------------------
# Data Browser column presets
# ---------------------------------------------------------------------------

# Data Browser column presets.
# These are UI conveniences only; the actual available columns come from Parquet schema discovery.
# Used by callbacks/data_browser_callbacks.py.
PRESET_COLUMN_GROUPS = {
    "ensembl_biotypes": [
        "accession", "species", "total_gene_biotypes",
        "protein_coding_count", # "protein_coding_percentage",
        "lncRNA_count", # "lncRNA_percentage",
        "rRNA_count", # "rRNA_percentage",
        "tRNA_count", # "tRNA_percentage",
        "miRNA_count" ,# "miRNA_percentage",
        "pseudogene_count", # "pseudogene_percentage",
        "processed_pseudogene_count", # "processed_pseudogene_percentage",
        "antisense_count", # "antisense_percentage",
        "snRNA_count", # "snRNA_percentage",
        "snoRNA_count", # "snoRNA_percentage",
        "gtf_file", "biodiversity_portal", "ensembl_browser"
    ],
    "ensembl_assembly": [
        "accession",
        "species",
        "ens_assembly_*",
    ],
    "ensembl_coding": [
        "accession",
        "species",
        "ens_coding_*",
    ],
    "ensembl_non_coding": [
        "accession",
        "species",
        "ens_non_coding_*",
    ],
    "ensembl_pseudogene": [
        "accession",
        "species",
        "ens_pseudogene_*",
    ],
    "ena_assembly_metrics": [
        "accession",
        "species",
        "ena_*",
    ],
    "taxonomy": ["accession", "tax_id", "kingdom", "phylum", "class", "order", "family", "genus", "species"],
    "bioclimate": ["accession", "species", "clim_bio1_mean", "clim_bio1_max", "clim_bio1_min", "clim_bio7_mean",
                   "clim_bio7_max", "clim_bio7_min", "clim_bio12_mean", "clim_bio12_max", "clim_bio12_min",
                   "clim_bio15_mean", "clim_bio15_max", "clim_bio15_min"],
    "distribution": ["accession", "species", "range_km2","mean_elevation","min_elevation", "max_elevation", "gbif"], # bioregions to add
    "sources": ["accession", "species", "biodiversity_portal", "gtf_file", "ensembl_browser", "gbif"]
}

DEFAULT_COLUMN_PRESET = "ensembl_biotypes"

# ---------------------------------------------------------------------------
# Gene biotype detection
# ---------------------------------------------------------------------------

# Gene biotype columns are discovered by suffix.
# For a base biotype like "lncRNA", the app expects:
#   lncRNA_count
#   lncRNA_percentage
# Used by utils.parquet_io.list_biotype_columns() to decide which *_count and
# *_percentage columns are true gene biotypes. This prevents ENA/Ensembl metrics
# such as ena_contig_count or ens_assembly_gc_percentage from appearing as
# biotypes in KPIs, filters, and plots.
GENE_BIOTYPE_PREFIX = ""
GENE_BIOTYPE_COUNT_SUFFIX = "_count"
GENE_BIOTYPE_PCT_SUFFIX = "_percentage"

GENE_BIOTYPE_BASES = {
    "protein_coding",
    "lncRNA",
    "rRNA",
    "tRNA",
    "miRNA",
    "pseudogene",
    "processed_pseudogene",
    "antisense",
    "snRNA",
    "snoRNA",
    "IG_C_gene",
    "IG_D_gene",
    "IG_J_gene",
    "IG_V_gene",
    "Mt_rRNA",
    "Mt_tRNA",
    "TR_C_gene",
    "TR_J_gene",
    "TR_V_gene",
    "Y_RNA",
    "misc_RNA",
    "ribozyme",
    "scaRNA",
    "transcribed_processed_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "unitary_pseudogene",
    "unprocessed_pseudogene",
    "vault_RNA",
}

# Columns to ignore when detecting biotypes
GENE_BIOTYPE_EXCLUDE = {"total_gene_biotypes"}

# ---------------------------------------------------------------------------
# Data Browser rendering and biotype normalization
# ---------------------------------------------------------------------------
# Data Browser renders these columns as clickable Markdown links.
URL_COLUMNS = ["biodiversity_portal", "gtf_file", "ensembl_browser", "gbif"]

# Total number of annotated genes per accession (used for normalization if present)
# Used as the denominator for biotype percentage and per-1k calculations when available.
TOTAL_GENES_COL = "total_gene_biotypes"

