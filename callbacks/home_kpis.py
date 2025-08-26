# callbacks/home_kpis.py
from __future__ import annotations

from typing import Dict, Sequence, Set

import pandas as pd
import pyarrow.dataset as ds
from dash import Input, Output, callback

from utils import config
from utils.parquet_io import (
    _dataset,                      # internal helpers — OK to use in callbacks
    _build_filter_expr,
    _build_biotype_pct_pushdown,
    _build_range_expr,
    list_biotype_columns,
    distinct_values_for_column,
    summarize_biotype_totals,
)

# --------------------------- formatting helpers ---------------------------

def _fmt_int(n) -> str:
    """Format integers with thousands separators; fallback to str."""
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


# --------------------------- accession helper ---------------------------

def _filtered_accessions(
    taxonomy_filter_map: Dict[str, Sequence[str]] | None,
    climate_filter: Sequence[str] | None,
    bio_levels_filter: Sequence[str] | None,
    bio_values_filter: Sequence[str] | None,
    biotype_pct_filter: Dict | None,
    climate_ranges=None, biogeo_ranges=None
) -> Set[str]:
    """
    Return the set of accessions from dashboard_main that survive:
    taxonomy + climate (+ biogeo via accession allow-list) + optional biotype% row filter.
    """
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not main or not config.ACCESSION_COL_MAIN:
        return set()

    # Base predicate: taxonomy/climate; biogeo added below
    expr = _build_filter_expr(
        dset=main,
        taxonomy_filter=None,
        taxonomy_filter_map=taxonomy_filter_map or {},
        climate_filter=climate_filter or [],
        accession_filter=None,
    )

    r1 = _build_range_expr(main, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(main, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    # Restrict to accessions that match selected biogeo level/value (if any)
    if bio_levels_filter or bio_values_filter:
        bset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
        if bset:
            filt = None
            if bio_levels_filter:
                filt = ds.field(config.BIOGEO_LEVEL_COL).isin([str(x) for x in bio_levels_filter])
            if bio_values_filter:
                f2 = ds.field(config.BIOGEO_VALUE_COL).isin([str(x) for x in bio_values_filter])
                filt = f2 if filt is None else (filt & f2)
            t = bset.to_table(columns=[config.ACCESSION_COL_BIOGEO], filter=filt)
            accs = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
            if accs:
                e_acc = ds.field(config.ACCESSION_COL_MAIN).isin(accs)
                expr = e_acc if expr is None else (expr & e_acc)

    # Try pushdown of biotype% if possible
    pct_expr, _pct_col = _build_biotype_pct_pushdown(main, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)
        tbl = main.to_table(columns=[config.ACCESSION_COL_MAIN], filter=expr)
        return set(pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique().tolist())

    # No pushdown → compute row mask from counts/denominator; project minimal columns
    read_cols: set[str] = {config.ACCESSION_COL_MAIN}
    target_cnt_col = None
    total_col = config.TOTAL_GENES_COL if (config.TOTAL_GENES_COL in main.schema.names) else None
    pmin = pmax = None

    if biotype_pct_filter and biotype_pct_filter.get("biotype"):
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        candidate = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if candidate in main.schema.names:
            target_cnt_col = candidate
            pmin = float(biotype_pct_filter.get("min", 0.0))
            pmax = float(biotype_pct_filter.get("max", 100.0))
            read_cols.add(candidate)
            if total_col:
                read_cols.add(total_col)
            else:
                col_map = list_biotype_columns()
                for c in col_map.get("count", []):
                    if c in main.schema.names:
                        read_cols.add(c)

    # If still no target column, we can just project the accessions
    if not target_cnt_col:
        tbl = main.to_table(columns=[config.ACCESSION_COL_MAIN], filter=expr)
        return set(pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique().tolist())

    accs: Set[str] = set()
    scanner = ds.Scanner.from_dataset(main, columns=list(read_cols), filter=expr, batch_size=8192)
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

        numer = pd.to_numeric(pdf[target_cnt_col], errors="coerce").astype(float)
        if total_col and total_col in pdf.columns:
            denom = pd.to_numeric(pdf[total_col], errors="coerce").astype(float)
        else:
            col_map = list_biotype_columns()
            cnt_cols = [c for c in col_map.get("count", []) if c in pdf.columns]
            denom = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce") for c in cnt_cols}).sum(axis=1).astype(float)

        valid = denom > 0
        pct = pd.Series(float("nan"), index=pdf.index, dtype="float64")
        pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

        mask = pct.ge(pmin).where(~pct.isna(), False) & pct.le(pmax).where(~pct.isna(), False)
        sub = pdf.loc[mask, config.ACCESSION_COL_MAIN].dropna().astype(str).unique().tolist()
        accs.update(sub)

    return accs


# --------------------------- KPIs callback ---------------------------

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
    geo_rng = gf.get("biogeo_ranges") or None

    # --- Taxonomy distinct counts (respecting ALL filters, incl. biotype%) ---
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

    # --- Biogeography distinct counts among filtered accessions (biotype% aware) ---
    accs = _filtered_accessions(tax_map, climate, bio_lvls, bio_vals, biopct,
                                clim_rng, geo_rng)
    realm_cnt = biome_cnt = ecoregion_cnt = 0
    if accs:
        bset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
        if bset:
            def _count_for_level(level_name: str) -> int:
                filt = (
                    ds.field(config.ACCESSION_COL_BIOGEO).isin(list(accs)) &
                    ds.field(config.BIOGEO_LEVEL_COL).isin([level_name])
                )
                tbl = bset.to_table(columns=[config.BIOGEO_VALUE_COL], filter=filt)
                ser = pd.Series(tbl.column(0).to_pandas()).dropna().astype(str)
                return int(ser.nunique())

            realm_cnt     = _count_for_level("realm")
            biome_cnt     = _count_for_level("biome")
            ecoregion_cnt = _count_for_level("ecoregion")

    # --- Total annotated genes (biotype% aware) ---
    total_genes = "-"
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if main and config.TOTAL_GENES_COL in main.schema.names:
        expr = _build_filter_expr(
            dset=main,
            taxonomy_filter=None,
            taxonomy_filter_map=tax_map,
            climate_filter=climate,
            accession_filter=list(accs) if accs else None,
        )

        pct_expr, _pct_col = _build_biotype_pct_pushdown(main, biopct)
        if pct_expr is not None:
            expr = pct_expr if expr is None else (expr & pct_expr)
            tbl = main.to_table(columns=[config.TOTAL_GENES_COL], filter=expr)
            s = pd.to_numeric(pd.Series(tbl.column(0).to_pandas()), errors="coerce").fillna(0)
            total_genes = _fmt_int(s.sum())
        else:
            # compute row mask then sum the total column for passing rows
            read_cols = {config.TOTAL_GENES_COL}
            target_cnt_col = None
            total_col = config.TOTAL_GENES_COL  # reuse as denominator if present
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
            total_sum = 0
            for rb in scanner.to_batches():
                if rb.num_rows == 0:
                    continue
                pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

                if target_cnt_col:
                    numer = pd.to_numeric(pdf[target_cnt_col], errors="coerce").astype(float)
                    if config.TOTAL_GENES_COL in pdf.columns:
                        denom = pd.to_numeric(pdf[config.TOTAL_GENES_COL], errors="coerce").astype(float)
                    else:
                        col_map = list_biotype_columns()
                        cnt_cols = [c for c in col_map.get("count", []) if c in pdf.columns]
                        denom = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce") for c in cnt_cols}).sum(axis=1).astype(float)

                    valid = denom > 0
                    pct = pd.Series(float("nan"), index=pdf.index, dtype="float64")
                    pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0
                    mask = pct.ge(pmin).where(~pct.isna(), False) & pct.le(pmax).where(~pct.isna(), False)

                    total_sum += pd.to_numeric(pdf[config.TOTAL_GENES_COL], errors="coerce").fillna(0).loc[mask].sum()
                else:
                    total_sum += pd.to_numeric(pdf[config.TOTAL_GENES_COL], errors="coerce").fillna(0).sum()

            total_genes = _fmt_int(total_sum)

    # --- Top biotypes by total count (already biotype% aware via summarize_biotype_totals) ---
    biotot = summarize_biotype_totals(
        taxonomy_filter_map=tax_map,
        climate_filter=climate,
        bio_levels_filter=bio_lvls,
        bio_values_filter=bio_vals,
        biotype_pct_filter=biopct,
        climate_ranges=clim_rng,
        biogeo_ranges=geo_rng,
    )
    if isinstance(biotot, pd.DataFrame) and not biotot.empty:
        top = biotot.nlargest(3, "count")
        desc = " • ".join([f"{r.biotype}: {_fmt_int(r.count)}" for r in top.itertuples()])
    else:
        desc = "No biotypes under current filters"

    # Map taxonomy counts into fixed 7 outputs (pad with zeros if needed)
    def _get(i):
        return _fmt_int(tax_counts[i]) if i < len(tax_counts) else "0"

    return (
        _get(0), _get(1), _get(2), _get(3), _get(4), _get(5), _get(6),
        _fmt_int(realm_cnt), _fmt_int(biome_cnt), _fmt_int(ecoregion_cnt),
        total_genes,
        desc,
    )
