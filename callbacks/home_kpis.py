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
    gf = gf or {}
    tax_map  = gf.get("taxonomy_map") or {}
    climate  = gf.get("climate") or []
    bio_lvls = gf.get("bio_levels") or []
    bio_vals = gf.get("bio_values") or []
    biopct   = gf.get("biotype_pct") or None
    clim_rng = gf.get("climate_ranges") or None
    geo_rng  = gf.get("biogeo_ranges") or None

    # Taxonomy distinct counts
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    tax_counts = []
    for col in ranks:
        vals = distinct_values_for_column(
            col,
            taxonomy_filter_map=tax_map,
            climate_filter=climate,
            bio_levels_filter=bio_lvls,
            bio_values_filter=bio_vals,
            biotype_pct_filter=biopct,
            climate_ranges=clim_rng,
            biogeo_ranges=geo_rng,
        )
        tax_counts.append(len(vals))
    while len(tax_counts) < 7:
        tax_counts.append(0)

    # Accessions under current filters
    accs = kpi_filtered_accessions(
        taxonomy_filter_map=tax_map,
        climate_filter=climate,
        bio_levels_filter=bio_lvls,
        bio_values_filter=bio_vals,
        biotype_pct_filter=biopct,
        climate_ranges=clim_rng,
        biogeo_ranges=geo_rng,
    )

    # Biogeography counts from biogeo_long
    realm_cnt, biome_cnt, ecoregion_cnt = kpi_biogeo_distinct_counts(accs)

    # Total genes (pushdown fast path; fallback to row-mask)
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    total_genes = "0"
    if main and config.TOTAL_GENES_COL in main.schema.names:
        expr = _build_filter_expr(
            dset=main,
            taxonomy_filter=None,
            taxonomy_filter_map=tax_map,
            climate_filter=climate,
            accession_filter=None,
        )
        r1 = _build_range_expr(main, clim_rng)
        if r1 is not None:
            expr = r1 if expr is None else (expr & r1)
        r2 = _build_range_expr(main, geo_rng)
        if r2 is not None:
            expr = r2 if expr is None else (expr & r2)
        if accs:
            e_acc = ds.field(config.ACCESSION_COL_MAIN).isin(list(accs))
            expr = e_acc if expr is None else (expr & e_acc)

        pct_expr, _pct_col = _build_biotype_pct_pushdown(main, biopct)
        if pct_expr is not None:
            expr = pct_expr if expr is None else (expr & pct_expr)
            tbl = main.to_table(columns=[config.TOTAL_GENES_COL], filter=expr)
            s = pd.to_numeric(pd.Series(tbl.column(0).to_pandas()), errors="coerce").fillna(0)
            total_genes = kpi_format_int(s.sum())
        else:
            # Fallback: compute row mask in pandas for biotype% range
            read_cols: set[str] = {config.TOTAL_GENES_COL}
            target_cnt_col = None
            total_col = config.TOTAL_GENES_COL
            pmin = pmax = None

            if biopct and biopct.get("biotype"):
                sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
                pref = config.GENE_BIOTYPE_PREFIX or ""
                candidate = f"{pref}{biopct['biotype']}{sfx}"
                if candidate in main.schema.names:
                    target_cnt_col = candidate
                    pmin = float(biopct.get("min", 0.0))
                    pmax = float(biopct.get("max", 100.0))
                    read_cols.add(candidate)
                    if config.TOTAL_GENES_COL not in main.schema.names:
                        col_map = list_biotype_columns()
                        for c in col_map.get("count", []):
                            if c in main.schema.names:
                                read_cols.add(c)

            scanner = ds.Scanner.from_dataset(main, columns=list(read_cols), filter=expr, batch_size=8192)
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
                    mask = pct.ge(pmin).where(~pct.isna(), False) & pct.le(pmax).where(~pct.isna(), False)
                    pdf = pdf.loc[mask]

                s = pd.to_numeric(pdf.get(config.TOTAL_GENES_COL), errors="coerce").fillna(0)
                total_genes = kpi_format_int(s.sum())

    # Top biotypes
    top = summarize_biotype_totals(
        taxonomy_filter_map=tax_map,
        climate_filter=climate,
        bio_levels_filter=bio_lvls,
        bio_values_filter=bio_vals,
        climate_ranges=clim_rng,
        biogeo_ranges=geo_rng,
    )

    if not top.empty:
        top = top.head(5)
        desc = " • ".join([f"{r.biotype}: {kpi_format_int(r.count)}" for r in top.itertuples()])
    else:
        desc = "No biotypes under current filters"

    def _get(i):
        return kpi_format_int(tax_counts[i]) if i < len(tax_counts) else "0"

    return (
        _get(0), _get(1), _get(2), _get(3), _get(4), _get(5), _get(6),
        kpi_format_int(realm_cnt), kpi_format_int(biome_cnt), kpi_format_int(ecoregion_cnt),
        total_genes,
        desc,
    )
