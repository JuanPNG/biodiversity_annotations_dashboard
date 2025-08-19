# utils/config.py
from __future__ import annotations
from pathlib import Path

# Where the parquet files live
DATA_DIR = Path("data/processed")

# Filenames (without hard-coding paths elsewhere)
DASHBOARD_MAIN_FN = "dashboard_main.parquet"
BIOGEO_LONG_FN = "biogeo_long.parquet"
GBIF_OCCURRENCES_FN = "gbif_occurrences.parquet"  # not used yet in filters

# === Tell the app your actual column names here ===
ACCESSION_COL_MAIN = "accession"         # in dashboard_main
TAXONOMY_COL = "species"
CLIMATE_LABEL_COL = "clim_bio1_mean"

# Biogeographic region label column in biogeo_long (e.g. "biogeographic_region")
ACCESSION_COL_BIOGEO = "accession"       # in biogeo_long
BIOGEO_LEVEL_COL = "level"
BIOGEO_VALUE_COL = "value"
BIOGEO_LEVELS_TO_EXPOSE = ["realm", "biome", "ecoregion"]

# Supports exact names and simple prefix wildcards like 'gene_biotype_*'
PRESET_COLUMN_GROUPS = {
    "genome": [
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
    "taxonomy": ["accession", "tax_id", "kingdom", "phylum", "class", "order", "family", "genus", "species"],
    "bioclimate": ["accession", "species", "clim_bio1_mean", "clim_bio1_max", "clim_bio1_min", "clim_bio7_mean",
                   "clim_bio7_max", "clim_bio7_min", "clim_bio12_mean", "clim_bio12_max", "clim_bio12_min",
                   "clim_bio15_mean", "clim_bio15_max", "clim_bio15_min"],
    "distribution": ["accession", "species", "range_km2","mean_elevation","min_elevation", "max_elevation", "gbif"], # bioregions to add
    "sources": ["accession", "species", "biodiversity_portal", "gtf_file", "ensembl_browser", "gbif"]
}

DEFAULT_COLUMN_PRESET = "genome"

# Gene biotype column naming
GENE_BIOTYPE_PREFIX = ""
GENE_BIOTYPE_COUNT_SUFFIX = "_count"
GENE_BIOTYPE_PCT_SUFFIX = "_percentage"

# Columns to ignore when detecting biotypes
GENE_BIOTYPE_EXCLUDE = {"total_gene_biotypes"}
# Chart defaults
BIOTYPE_TOP_N_DEFAULT = 15

# Columns that contain URLs (rendered as clickable links in the grid)
URL_COLUMNS = ["biodiversity_portal", "gtf_file", "ensembl_browser", "gbif"]

