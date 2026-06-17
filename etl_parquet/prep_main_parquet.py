"""Build the dashboard's accession-level Parquet table.

This module flattens selected nested fields from the source integrated Parquet
into one row per accession. Repeated or categorical data that is not naturally
one-to-one with accession should stay in separate ETL outputs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence
import math
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


GENE_BIOTYPE_KEYS: Sequence[str] = (
    # 10 most relevant
    "protein_coding","lncRNA","rRNA","tRNA","miRNA",
    "pseudogene","processed_pseudogene","antisense","snRNA","snoRNA",
    # Others
    "IG_C_gene", "IG_D_gene", "IG_J_gene", "IG_V_gene",
    "Mt_rRNA", "Mt_tRNA",
    "TR_C_gene", "TR_J_gene", "TR_V_gene",
    "Y_RNA", "misc_RNA",
    "ribozyme", "scaRNA",
    "transcribed_processed_pseudogene", "transcribed_unprocessed_pseudogene",
    "unitary_pseudogene", "unprocessed_pseudogene", "vault_RNA"
)
CLIMATE_VARS: Sequence[str] = ("bio1","bio7","bio12","bio15")
CLIMATE_STATS: Sequence[str] = ("mean","max","min")

# Accession-level assembly metadata from source column `ena_stats`.
# Output columns use the pattern `ena_<field>`.
ENA_STATS_KEYS: Sequence[str] = (
    "assembly_level",
    "ungapped_length",
    "scaffold_n50",
    "scaffold_count",
    "contig_n50",
    "contig_count",
    "coverage",
    "spanned_gaps",
    "unspanned_gaps",
    "contig_l50",
    "scaffold_l50",
    "contig_n75",
    "contig_n90",
    "scaffold_n75",
    "scaffold_n90",
    "replicon_count",
    "non_chromosome_replicon_count",
)

# Accession-level Ensembl statistics from source column `ensembl_stats`.
# Only the groups listed here are exported. Output columns use:
# `ens_<group>_<field>`, e.g. `ens_coding_total_transcripts`.
ENSEMBL_STATS_GROUPS: dict[str, Sequence[str]] = {
    "assembly": (
        "contig_n50",
        "total_genome_length",
        "total_coding_sequence_length",
        "total_gap_length",
        "spanned_gaps",
        "chromosomes",
        "toplevel_sequences",
        "component_sequences",
        "gc_percentage",
    ),
    "coding": (
        "coding_genes",
        "average_genomic_span",
        "average_sequence_length",
        "average_cds_length",
        "shortest_gene_length",
        "longest_gene_length",
        "total_transcripts",
        "coding_transcripts",
        "transcripts_per_gene",
        "coding_transcripts_per_gene",
        "total_exons",
        "total_coding_exons",
        "average_exon_length",
        "average_coding_exon_length",
        "average_exons_per_transcript",
        "average_coding_exons_per_coding_transcript",
        "total_introns",
        "average_intron_length",
    ),
    "non_coding": (
        "non_coding_genes",
        "small_non_coding_genes",
        "long_non_coding_genes",
        "misc_non_coding_genes",
        "average_genomic_span",
        "average_sequence_length",
        "shortest_gene_length",
        "longest_gene_length",
        "total_transcripts",
        "transcripts_per_gene",
        "total_exons",
        "average_exon_length",
        "average_exons_per_transcript",
        "total_introns",
        "average_intron_length",
    ),
    "pseudogene": (
        "pseudogenes",
        "average_genomic_span",
        "average_sequence_length",
        "shortest_gene_length",
        "longest_gene_length",
        "total_transcripts",
        "transcripts_per_gene",
        "total_exons",
        "average_exon_length",
        "average_exons_per_transcript",
        "total_introns",
        "average_intron_length",
    ),
}

# PARQUET_PATH = '../data/original/integ_genome_features_20250812.parquet'


def _safe_col(table: pa.Table, name: str) -> List[Any]:
    """Return column as a list, or a list of Nones if missing."""
    return table[name].to_pylist() if name in table.column_names else [None] * len(table)


def _prefixed_values(
    data: Any,
    prefix: str,
    keys: Sequence[str],
) -> Dict[str, Any]:
    """Return selected struct values as flat, prefixed columns.

    Missing structs are represented with explicit None values so every output
    row has the same accession-level schema.
    """
    if not isinstance(data, dict):
        return {f"{prefix}_{key}": None for key in keys}

    return {f"{prefix}_{key}": data.get(key) for key in keys}


def _flatten_ensembl_stats(data: Any) -> Dict[str, Any]:
    """Flatten the supported nested Ensembl stats groups for one accession."""
    out: Dict[str, Any] = {}

    if not isinstance(data, dict):
        for group, keys in ENSEMBL_STATS_GROUPS.items():
            for key in keys:
                out[f"ens_{group}_{key}"] = None
        return out

    source_group_names = {
        "assembly": "assembly_stats",
        "coding": "coding_stats",
        "non_coding": "non_coding_stats",
        "pseudogene": "pseudogene_stats",
    }

    for group, keys in ENSEMBL_STATS_GROUPS.items():
        source_group = data.get(source_group_names[group])
        out.update(_prefixed_values(source_group, f"ens_{group}", keys))

    return out


def _first_struct(data: Any) -> Dict[str, Any] | None:
    """Return a struct dict from either current struct or legacy list<struct> input."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def build_main_table(parquet_path: str) -> pd.DataFrame:
    """Build the wide accession-level table consumed by the dashboard.

    The output keeps one row per accession and includes taxonomy, climate,
    distribution, source URLs, gene biotype summaries, and selected ENA/Ensembl
    assembly and annotation statistics.
    """
    wanted = [
        "accession", "taxonomy", "gene_biotypes", "clim_CHELSA", "meta_urls",
        "range_km2", "mean_elevation", "min_elevation", "max_elevation", "median_elevation",
        "ena_stats", "ensembl_stats",
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
    ena_col = _safe_col(table, "ena_stats")
    ens_col = _safe_col(table, "ensembl_stats")

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

        # Flatten one-to-one accession stats from nested source structs.
        out.update(_prefixed_values(ena_col[i], "ena", ENA_STATS_KEYS))
        out.update(_flatten_ensembl_stats(ens_col[i]))

        # Taxonomy
        tax = _first_struct(tax_col[i])
        if tax is not None:
            for k in ("kingdom", "phylum", "class", "order", "family", "genus", "species"):
                out[k] = tax.get(k)
            out["tax_id"] = tax.get("tax_id")

        # Climate
        clim = clim_col[i]
        if isinstance(clim, dict):
            for var in CLIMATE_VARS:
                vb = clim.get(var)
                if isinstance(vb, dict):
                    for stat in CLIMATE_STATS:
                        out[f"clim_{var}_{stat}"] = vb.get(stat)

        # URLs
        mu = _first_struct(urls_col[i])
        if mu is not None:
            out["biodiversity_portal"] = mu.get("Biodiversity_portal")
            out["gtf_file"] = mu.get("GTF")
            out["ensembl_browser"] = mu.get("Ensembl_browser")
            out["gbif"] = mu.get("GBIF")

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

    stats_cols = (
        [f"ena_{key}" for key in ENA_STATS_KEYS] +
        [
            f"ens_{group}_{key}"
            for group, keys in ENSEMBL_STATS_GROUPS.items()
            for key in keys
        ]
    )

    ordered = (
        [c for c in front if c in df.columns] +
        [c for c in biotype_cols if c in df.columns] +
        [c for c in stats_cols if c in df.columns] +
        [c for c in df.columns if c not in set(front + biotype_cols + stats_cols)]
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
