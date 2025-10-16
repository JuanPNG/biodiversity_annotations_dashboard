from __future__ import annotations

import pandas as pd
import pyarrow.dataset as ds
from dash import Input, Output, callback

from utils import config
from utils.parquet_io import (
    _dataset,
    _build_filter_expr,
    _build_biotype_pct_pushdown,
    _build_range_expr,
    list_biotype_columns,
    distinct_values_for_column,
    summarize_biotype_totals,
)
from utils.data_tools import (
    kpi_format_int,
    kpi_filtered_accessions,
    kpi_biogeo_distinct_counts,
)


@callback(
    Output("kpi-kingdom", "children"),
    Output("kpi-phylum", "children"),
    Output("kpi-class", "children"),
    Output("kpi-order", "children"),
    Output("kpi-family", "children"),
    Output("kpi-genus", "children"),
    Output("kpi-species", "children"),
    Output("kpi-bio-realm", "children"),
    Output("kpi-bio-biome", "children"),
    Output("kpi-bio-ecoregion", "children"),
    Output("kpi-total-genes", "children"),
    Output("kpi-top-biotypes", "children"),
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
def update_home_kpis(gf):
    """Compute and update Home page KPIs using the active global filters.
    When no filters are provided, compute unfiltered KPIs.
    """
    gf = gf or {}
    tax_map = gf.get("taxonomy_map") or {}
    climate_labels = gf.get("climate") or []
    bio_levels = gf.get("bio_levels") or []
    bio_values = gf.get("bio_values") or []
    biotype_pct = gf.get("biotype_pct") or None
    climate_ranges = gf.get("climate_ranges") or None
    biogeo_ranges = gf.get("biogeo_ranges") or None

    # --- Taxonomy distinct counts (per configured rank) ---
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    tax_counts: list[int] = []
    for col in ranks:
        distinct_vals = distinct_values_for_column(
            col,
            taxonomy_filter_map=tax_map,
            climate_filter=climate_labels,
            bio_levels_filter=bio_levels,
            bio_values_filter=bio_values,
            biotype_pct_filter=biotype_pct,
            climate_ranges=climate_ranges,
            biogeo_ranges=biogeo_ranges,
        )
        tax_counts.append(len(distinct_vals))
    while len(tax_counts) < 7:
        tax_counts.append(0)

    # --- Accessions under current filters (for biogeo distincts) ---
    accession_ids = kpi_filtered_accessions(
        taxonomy_filter_map=tax_map,
        climate_filter=climate_labels,
        bio_levels_filter=bio_levels,
        bio_values_filter=bio_values,
        biotype_pct_filter=biotype_pct,
        climate_ranges=climate_ranges,
        biogeo_ranges=biogeo_ranges,
    )

    # --- Biogeography counts from biogeo_long ---
    realm_cnt, biome_cnt, ecoregion_cnt = kpi_biogeo_distinct_counts(accession_ids)

    # --- Total genes (pushdown fast path; fallback to row-mask) ---
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    total_genes = "0"
    if main and config.TOTAL_GENES_COL in main.schema.names:
        # Base predicate from taxonomy/climate (no accession list yet)
        filter_expr = _build_filter_expr(
            dset=main,
            taxonomy_filter=None,
            taxonomy_filter_map=tax_map,
            climate_filter=climate_labels,
            accession_filter=None,
        )

        # Add numeric range predicates
        climate_range_expr = _build_range_expr(main, climate_ranges)
        if climate_range_expr is not None:
            filter_expr = climate_range_expr if filter_expr is None else (filter_expr & climate_range_expr)

        biogeo_range_expr = _build_range_expr(main, biogeo_ranges)
        if biogeo_range_expr is not None:
            filter_expr = biogeo_range_expr if filter_expr is None else (filter_expr & biogeo_range_expr)

        # Add accession allow-list (if any)
        if accession_ids:
            accession_filter_expr = ds.field(config.ACCESSION_COL_MAIN).isin(list(accession_ids))
            filter_expr = accession_filter_expr if filter_expr is None else (filter_expr & accession_filter_expr)

        # Try pushdown for biotype %; otherwise fallback to in-batch mask
        pct_pushdown_expr, _pct_col = _build_biotype_pct_pushdown(main, biotype_pct)
        if pct_pushdown_expr is not None:
            filter_expr = pct_pushdown_expr if filter_expr is None else (filter_expr & pct_pushdown_expr)
            tbl = main.to_table(columns=[config.TOTAL_GENES_COL], filter=filter_expr)
            total_genes_series = pd.to_numeric(
                pd.Series(tbl.column(0).to_pandas()), errors="coerce"
            ).fillna(0)
            total_genes = kpi_format_int(total_genes_series.sum())
        else:
            # Fallback: compute row mask in pandas for biotype% range
            read_cols: set[str] = {config.TOTAL_GENES_COL}
            target_cnt_col = None
            total_col = config.TOTAL_GENES_COL
            pct_min = pct_max = None

            if biotype_pct and biotype_pct.get("biotype"):
                sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
                pref = config.GENE_BIOTYPE_PREFIX or ""
                candidate = f"{pref}{biotype_pct['biotype']}{sfx}"
                if candidate in main.schema.names:
                    target_cnt_col = candidate
                    pct_min = float(biotype_pct.get("min", 0.0))
                    pct_max = float(biotype_pct.get("max", 100.0))
                    read_cols.add(candidate)
                    if config.TOTAL_GENES_COL not in main.schema.names:
                        col_map = list_biotype_columns()
                        for c in col_map.get("count", []):
                            if c in main.schema.names:
                                read_cols.add(c)

            scanner = ds.Scanner.from_dataset(main, columns=list(read_cols), filter=filter_expr, batch_size=8192)
            batches = scanner.to_batches()

            # Guard: if no data passed the filters, return 0 safely
            non_empty = [rb for rb in batches or [] if getattr(rb, "num_rows", 0) > 0]
            if not non_empty:
                total_genes = "0"
            else:
                pdf = pd.concat([rb.to_pandas() for rb in non_empty], ignore_index=True)

                if target_cnt_col:
                    numer = pd.to_numeric(pdf.get(target_cnt_col), errors="coerce")
                    if total_col and total_col in pdf.columns:
                        denom = pd.to_numeric(pdf[total_col], errors="coerce")
                    else:
                        col_map = list_biotype_columns()
                        cnt_cols = [c for c in (col_map.get("count", []) or []) if c in pdf.columns]
                        denom = pd.to_numeric(pdf[cnt_cols], errors="coerce").sum(axis=1) if cnt_cols else pd.Series(
                            0.0, index=pdf.index)
                    pct = pd.Series(pd.NA, index=pdf.index, dtype="float64")
                    valid = (denom > 0) & numer.notna()
                    pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0
                    mask = pct.ge(pct_min).where(~pct.isna(), False) & pct.le(pct_max).where(~pct.isna(), False)
                    pdf = pdf.loc[mask]

                total_genes_series = pd.to_numeric(pdf.get(config.TOTAL_GENES_COL), errors="coerce").fillna(0)
                total_genes = kpi_format_int(total_genes_series.sum())

    # --- Top biotypes summary (top 5 pretty string) ---
    top = summarize_biotype_totals(
        taxonomy_filter_map=tax_map,
        climate_filter=climate_labels,
        bio_levels_filter=bio_levels,
        bio_values_filter=bio_values,
        climate_ranges=climate_ranges,
        biogeo_ranges=biogeo_ranges,
    )
    if not top.empty:
        top = top.head(5)
        desc = " • ".join([f"{r.biotype}: {kpi_format_int(r.count)}" for r in top.itertuples()])
    else:
        desc = "No biotypes under current filters"

    def _get(i: int) -> str:
        return kpi_format_int(tax_counts[i]) if i < len(tax_counts) else "0"

    return (
        _get(0), _get(1), _get(2), _get(3), _get(4), _get(5), _get(6),
        kpi_format_int(realm_cnt), kpi_format_int(biome_cnt), kpi_format_int(ecoregion_cnt),
        total_genes,
        desc,
    )
