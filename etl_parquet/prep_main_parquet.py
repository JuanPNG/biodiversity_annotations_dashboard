from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence
import math
import pandas as pd
import pyarrow.parquet as pq


GENE_BIOTYPE_KEYS: Sequence[str] = (
    "protein_coding","lncRNA","rRNA","tRNA","miRNA",
    "pseudogene","processed_pseudogene","antisense","snRNA","snoRNA",
)
CLIMATE_VARS: Sequence[str] = ("bio1","bio7","bio12","bio15")
CLIMATE_STATS: Sequence[str] = ("mean","max","min")

# PARQUET_PATH = '../data/integ_genome_features_20250812.parquet'


def _safe_col(table: pq.Table, name: str) -> List[Any]:
    """Return column as a list, or a list of Nones if missing."""
    return table[name].to_pylist() if name in table.column_names else [None] * len(table)


def build_main_table(parquet_path: str) -> pd.DataFrame:
    wanted = [
        "accession","taxonomy","gene_biotypes","clim_CHELSA","meta_urls",
        "range_km2","mean_elevation","min_elevation","max_elevation","median_elevation",
    ]
    schema = pq.read_schema(parquet_path)
    cols = [c for c in wanted if c in schema.names]
    table = pq.read_table(parquet_path, columns=cols)

    # Extracting cols as lists to facilitate access
    acc      = _safe_col(table, "accession")
    tax_col  = _safe_col(table, "taxonomy")
    gene_col = _safe_col(table, "gene_biotypes")
    clim_col = _safe_col(table, "clim_CHELSA")
    urls_col = _safe_col(table, "meta_urls")

    rng          = _safe_col(table, "range_km2")
    elev_mean    = _safe_col(table, "mean_elevation")
    elev_min     = _safe_col(table, "min_elevation")
    elev_max     = _safe_col(table, "max_elevation")
    elev_median  = _safe_col(table, "median_elevation")

    #Building rows
    rows: List[Dict[str, Any]] = []

    for i in range(len(table)):
        out: Dict[str, Any] = {
            "accession":        acc[i],
            "range_km2":        rng[i],
            "mean_elevation":   elev_mean[i],
            "min_elevation":    elev_min[i],
            "max_elevation":    elev_max[i],
            "median_elevation": elev_median[i],
        }

        # Taxonomy
        tax_first = tax_col[i][0] if isinstance(tax_col[i], list) and tax_col[i] else None
        if isinstance(tax_first, dict):
            for k in ("kingdom","phylum","class","order","family","genus","species"):
                out[k] = tax_first.get(k)
            out["tax_id"] = tax_first.get("tax_id")

        # Climate
        clim = clim_col[i]
        if isinstance(clim, dict):
            for var in CLIMATE_VARS:
                vb = clim.get(var)
                if isinstance(vb, dict):
                    for stat in CLIMATE_STATS:
                        out[f"clim_{var}_{stat}"] = vb.get(stat)

        # URLs
        mu = urls_col[i]
        mu_first = mu[0] if isinstance(mu, list) and mu else None
        if isinstance(mu_first, dict):
            out["biodiversity_portal"] = mu_first.get("Biodiversity_portal")
            out["gtf_file"]            = mu_first.get("GTF")
            out["ensembl_browser"]     = mu_first.get("Ensembl_browser")
            out["gbif"]                = mu_first.get("GBIF")

        # Gene biotypes
        for k in GENE_BIOTYPE_KEYS:
            out[f"{k}_count"] = pd.NA
            out[f"{k}_percentage"] = math.nan
        out["total_gene_biotypes"] = pd.NA

        glist = gene_col[i] if isinstance(gene_col[i], list) else None
        if glist:
            for r in glist:
                if not isinstance(r, dict):
                    continue
                key = r.get("gene_biotype")
                if key in GENE_BIOTYPE_KEYS:
                    out[f"{key}_count"] = r.get("gene_biotype_count")
                    out[f"{key}_percentage"] = r.get("gene_biotype_percentage")
                if (out["total_gene_biotypes"] is pd.NA) and isinstance(r.get("total_gene_biotypes"), int):
                    out["total_gene_biotypes"] = r["total_gene_biotypes"]

        rows.append(out)

    df = pd.DataFrame(rows)

    # Dtypes
    for k in GENE_BIOTYPE_KEYS:
        c_count = f"{k}_count"; c_pct = f"{k}_percentage"
        if c_count in df: df[c_count] = df[c_count].astype("Int64")
        if c_pct   in df: df[c_pct]   = df[c_pct].astype("float32")
    if "total_gene_biotypes" in df:
        df["total_gene_biotypes"] = df["total_gene_biotypes"].astype("Int64")

    front = [
        "accession",
        "kingdom","phylum","class","order","family","genus","species","tax_id",
        "range_km2","mean_elevation","min_elevation","max_elevation","median_elevation",
        "clim_bio1_mean","clim_bio1_max","clim_bio1_min",
        "clim_bio7_mean","clim_bio7_max","clim_bio7_min",
        "clim_bio12_mean","clim_bio12_max","clim_bio12_min",
        "clim_bio15_mean", "clim_bio15_max", "clim_bio15_min",
        "biodiversity_portal","gtf_file","ensembl_browser","gbif",
        "total_gene_biotypes",
    ]
    biotype_cols = [c for k in GENE_BIOTYPE_KEYS for c in (f"{k}_count", f"{k}_percentage")]
    ordered = (
        [c for c in front if c in df.columns] +
        [c for c in biotype_cols if c in df.columns] +
        [c for c in df.columns if c not in set(front + biotype_cols)]
    )
    return df[ordered]


# df = build_main_table(parquet_path=PARQUET_PATH)
#
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.head())
#
# ja = df[df['accession'] == 'GCA_905220365.2'].to_dict('records')
# print(json.dumps(ja, indent=2))