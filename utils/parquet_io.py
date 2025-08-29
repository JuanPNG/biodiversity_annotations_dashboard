# -----------------------------------------------------------------------------
# Centralized Arrow/Parquet I/O helpers for the dashboard.
# - Column/domain discovery
# - Min/max extents for sliders (climate, range_km2)
# - Biotype column lists (pct/count)
# Keep functions pure/deterministic; avoid side effects and keep caching tight.
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from utils import config

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.types as pat
import pyarrow.dataset as ds

# Light simple alias for filter expressions.
DatasetExpr = Any

def _dataset(path: Path) -> ds.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return ds.dataset(str(path))

def list_columns(path: Path) -> list[str]:
    dset = _dataset(path)
    return list(dset.schema.names)

def pick_default_columns(all_cols: Sequence[str], max_cols: int = 25) -> list[str]:
    return list(all_cols[:max_cols])

def _get_accessions_for_biogeo(levels: Sequence[str] | None, values: Sequence[str] | None) -> set[str]:
    if not levels and not values:
        return set()
    bset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    expr = None
    if levels:
        expr = ds.field(config.BIOGEO_LEVEL_COL).isin([str(x) for x in levels])
    if values:
        e2 = ds.field(config.BIOGEO_VALUE_COL).isin([str(x) for x in values])
        expr = e2 if expr is None else (expr & e2)
    table = bset.to_table(columns=[config.ACCESSION_COL_BIOGEO], filter=expr)
    s = pd.Series(table.column(0).to_pandas()).dropna().astype(str)
    return set(s.unique())

def get_biogeo_tags_for_accessions_by_level(accessions: Sequence[str], levels: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    if not accessions or not levels:
        return {}
    bset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    cols = [config.ACCESSION_COL_BIOGEO, config.BIOGEO_LEVEL_COL, config.BIOGEO_VALUE_COL]
    expr = ds.field(config.ACCESSION_COL_BIOGEO).isin([str(a) for a in accessions]) & \
           ds.field(config.BIOGEO_LEVEL_COL).isin([str(l) for l in levels])
    table = bset.to_table(columns=cols, filter=expr)
    pdf = table.to_pandas()
    if pdf.empty:
        return {}
    pdf[config.ACCESSION_COL_BIOGEO] = pdf[config.ACCESSION_COL_BIOGEO].astype(str)
    out: dict[str, dict[str, list[str]]] = {}
    for acc, g_acc in pdf.groupby(config.ACCESSION_COL_BIOGEO):
        slot: dict[str, list[str]] = {}
        for lvl, g_lvl in g_acc.groupby(config.BIOGEO_LEVEL_COL):
            vals = g_lvl[config.BIOGEO_VALUE_COL].dropna().astype(str).unique().tolist()
            slot[str(lvl)] = sorted(vals)
        out[acc] = slot
    return out

def _build_filter_expr(
    dset: ds.Dataset,
    taxonomy_filter: Sequence[str] | None,
    climate_filter: Sequence[str] | None,
    accession_filter: Sequence[str] | None,
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
) -> DatasetExpr | None:
    """
    Combine equality filters into a single dataset expression (AND).
    Supports multi-rank taxonomy via taxonomy_filter_map.
    """
    expr = None

    # Preferred: multi-rank taxonomy map (e.g., {"family": [...], "genus":[...]})
    if taxonomy_filter_map:
        for col, vals in taxonomy_filter_map.items():
            coerced = _coerce_values_for_column(dset, col, vals)
            if coerced:
                e = ds.field(col).isin(coerced)
                expr = e if expr is None else (expr & e)

    # Legacy single taxonomy column (only used if provided & configured)
    if (not taxonomy_filter_map) and taxonomy_filter and config.TAXONOMY_COL:
        coerced = _coerce_values_for_column(dset, config.TAXONOMY_COL, taxonomy_filter)
        if coerced:
            e = ds.field(config.TAXONOMY_COL).isin(coerced)
            expr = e if expr is None else (expr & e)

    # Optional legacy categorical climate (yours is None, so this will be skipped)
    if climate_filter and config.CLIMATE_LABEL_COL:
        coerced = _coerce_values_for_column(dset, config.CLIMATE_LABEL_COL, climate_filter)
        if coerced:
            e = ds.field(config.CLIMATE_LABEL_COL).isin(coerced)
            expr = e if expr is None else (expr & e)

    if accession_filter and config.ACCESSION_COL_MAIN:
        e = ds.field(config.ACCESSION_COL_MAIN).isin(list(accession_filter))
        expr = e if expr is None else (expr & e)

    return expr

def _build_biotype_pct_pushdown(
        dset: ds.Dataset,
        biotype_pct_filter: dict | None
) -> tuple[ds.Expression | None, str | None]:
    """
    If the chosen biotype has a precomputed *_percentage column, return (expr, column_name)
    where expr is: pct_col BETWEEN [min, max]. Otherwise (None, None).
    """
    if not biotype_pct_filter or not biotype_pct_filter.get("biotype"):
        return None, None
    pct_sfx = config.GENE_BIOTYPE_PCT_SUFFIX or "_percentage"
    pref = config.GENE_BIOTYPE_PREFIX or ""
    col = f"{pref}{biotype_pct_filter['biotype']}{pct_sfx}"
    if col in dset.schema.names:
        pmin = float(biotype_pct_filter.get("min", 0.0))
        pmax = float(biotype_pct_filter.get("max", 100.0))
        expr = (ds.field(col) >= pmin) & (ds.field(col) <= pmax)
        return expr, col
    return None, None

def _build_range_expr(
        dset: ds.Dataset,
        ranges: dict | None
) -> ds.Expression | None:
    """Build (col BETWEEN lo AND hi) AND ... for a dict like {'col': [lo, hi]}."""
    if not ranges:
        return None
    expr = None
    for col, pair in ranges.items():
        if col not in dset.schema.names:
            continue
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except Exception:
            continue
        e = (ds.field(col) >= lo) & (ds.field(col) <= hi)
        expr = e if expr is None else (expr & e)
    return expr

def list_biotype_columns() -> dict[str, list[str]]:
    """
    Return {'pct': [...], 'count': [...]} from dashboard_main based on *_percentage and *_count.
    Excludes items in config.GENE_BIOTYPE_EXCLUDE.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    cols = list_columns(main_path)

    pct_sfx = config.GENE_BIOTYPE_PCT_SUFFIX or "_percentage"
    cnt_sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
    exclude = set(config.GENE_BIOTYPE_EXCLUDE or ())

    pct = [c for c in cols if c.endswith(pct_sfx) and c not in exclude]
    cnt = [c for c in cols if c.endswith(cnt_sfx) and c not in exclude]
    return {"pct": pct, "count": cnt}


def summarize_biotypes(
    *,
    metric: str,  # "pct" (mean of *_percentage) or "count" (sum of *_count)
    biotype_cols: list[str] | None,
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
    taxonomy_filter: Sequence[str] | None = None,   # legacy (ignored if map is given)
    climate_filter: Sequence[str] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Efficiently summarize gene biotypes over ALL matching rows, respecting global filters.
    Returns DataFrame with columns ['biotype','value'].
    """
    assert metric in ("pct", "count")
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve accession filter from biogeo level/value
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return pd.DataFrame(columns=["biotype", "value"])
        accession_filter = list(accs)

    # Build predicate (type-aware, AND across ranks)
    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
    )

    # Which columns to read
    available = set(list_columns(main_path))
    if not biotype_cols:
        col_map = list_biotype_columns()
        cols = col_map["pct"] if metric == "pct" else col_map["count"]
    else:
        cols = [c for c in biotype_cols if c in available]
    if not cols:
        return pd.DataFrame(columns=["biotype", "value"])

    # Aggregate by streaming
    sums: dict[str, float] = {c: 0.0 for c in cols}
    counts: dict[str, int] = {c: 0 for c in cols}  # for mean of percentages

    scanner = ds.Scanner.from_dataset(dset, columns=cols, filter=expr, batch_size=batch_size)
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)
        for c in cols:
            s = pd.to_numeric(pdf[c], errors="coerce")
            if metric == "pct":
                v = s.dropna()
                sums[c] += float(v.sum())
                counts[c] += int(v.shape[0])
            else:
                sums[c] += float(s.fillna(0).sum())

    # Build tidy result
    pct_sfx = config.GENE_BIOTYPE_PCT_SUFFIX or "_percentage"
    cnt_sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"

    rows = []
    for c in cols:
        if metric == "pct" and c.endswith(pct_sfx):
            name = c[: -len(pct_sfx)]
            denom = max(1, counts[c])
            val = (sums[c] / denom) if denom else 0.0
        elif metric == "count" and c.endswith(cnt_sfx):
            name = c[: -len(cnt_sfx)]
            val = sums[c]
        else:
            name = c
            val = sums[c]
        rows.append({"biotype": name, "value": float(val)})

    df = pd.DataFrame(rows).dropna(subset=["value"])
    return df.sort_values("value", ascending=False, kind="mergesort").reset_index(drop=True)

def load_dashboard_page(
    *,
    columns: list[str],
    page: int,
    page_size: int,
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    climate_filter: dict | None = None,           # reserved for later
    biotype_pct_filter: dict | None = None,       # {"biotype": str, "min": float, "max": float}
    climate_ranges: dict | None = None,
    biogeo_ranges: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Return (page_df, total_rows_after_filters) for the Data Browser.

    Notes
    -----
    - Applies taxonomy + biogeo via Arrow predicate pushdown.
    - Applies optional biotype % row filter in-batch using pandas:
        pct(col) = biotype_count / total_gene_biotypes * 100 (fallback: sum of *_count)
    - Reads only requested `columns`, plus any extra columns needed to evaluate the pct filter/sort.
    - Performs server-side sort (if provided) after all filters, then slices the requested page.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve accession filter from biogeo
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return pd.DataFrame(columns=columns), 0
        accession_filter = list(accs)

    # Build predicate for pushdown (taxonomy + accession + (climate later))
    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=None,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=None,  # sliders later
        accession_filter=accession_filter,
    )

    # Add numeric range pushdown
    r1 = _build_range_expr(dset, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(dset, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    pct_expr, pct_col = _build_biotype_pct_pushdown(dset, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    # Build the read column set
    read_cols: set[str] = set(columns)
    if pct_col:
        read_cols.add(pct_col)

    # --- FALLBACK: if no % col, compute from counts in-batch ---
    target_cnt_col = None
    total_col = config.TOTAL_GENES_COL if (config.TOTAL_GENES_COL in dset.schema.names) else None
    if biotype_pct_filter and biotype_pct_filter.get("biotype"):
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        candidate = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if candidate in dset.schema.names:
            target_cnt_col = candidate
            read_cols.add(candidate)
        if total_col:
            read_cols.add(total_col)
        else:
            # Fallback: we need all *_count columns to compute per-row total
            col_map = list_biotype_columns()
            for c in col_map.get("count", []):
                if c in dset.schema.names:
                    read_cols.add(c)

    # Scanner
    scanner = ds.Scanner.from_dataset(
        dset,
        columns=list(read_cols),
        filter=expr,
        batch_size=8192,
    )

    # Collect filtered rows
    frames: list[pd.DataFrame] = []
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

        # Apply biotype % mask if needed
        if target_cnt_col:
            numer = pd.to_numeric(pdf[target_cnt_col], errors="coerce").astype(float)
            if total_col and total_col in pdf.columns:
                denom = pd.to_numeric(pdf[total_col], errors="coerce").astype(float)
            else:
                # sum of all *_count columns present in this batch
                col_map = list_biotype_columns()
                cnt_cols = [c for c in col_map.get("count", []) if c in pdf.columns]
                denom = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce") for c in cnt_cols}).sum(axis=1).astype(float)

            valid = denom > 0
            pct = pd.Series(np.nan, index=pdf.index, dtype="float64")
            pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

            pmin = float(biotype_pct_filter.get("min", 0.0))
            pmax = float(biotype_pct_filter.get("max", 100.0))
            mask = pct.ge(pmin) & pct.le(pmax)
            mask = mask.fillna(False)
            pdf = pdf[mask]
            if pdf.empty:
                continue

        # Keep only requested display columns in the same order
        frames.append(pdf[list(columns)])

    if not frames:
        return pd.DataFrame(columns=columns), 0

    all_rows = pd.concat(frames, ignore_index=True)

    # Page slice
    start = max(0, (max(1, int(page)) - 1) * int(page_size))
    end = start + int(page_size)
    page_df = all_rows.iloc[start:end].reset_index(drop=True)

    return page_df, int(len(page_df))



def count_dashboard_rows(
    *,
    taxonomy_filter: Sequence[str] | None = None,
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
    climate_filter: Sequence[str] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    biotype_pct_filter: dict | None = None,     # {"biotype": str, "min": float, "max": float}
    climate_ranges: dict | None = None,
    biogeo_ranges: dict | None = None,
) -> int:
    """
        Count rows after applying taxonomy + biogeo pushdown and optional biotype-% row filter.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve accession filter from biogeo
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return 0
        accession_filter = list(accs)

    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
        taxonomy_filter_map=taxonomy_filter_map,
    )

    # Add numeric ranges for clim and biogeo
    r1 = _build_range_expr(dset, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(dset, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    # --- try percentage pushdown first ---
    pct_expr, pct_col = _build_biotype_pct_pushdown(dset, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    # If pushdown exists, counting is trivial
    if pct_expr is not None:
        first_col = dset.schema.names[0]
        scanner = ds.Scanner.from_dataset(dset, columns=[first_col], filter=expr, batch_size=8192)
        total = 0
        for rb in scanner.to_batches():
            total += int(rb.num_rows)
        return total

    # Minimal column set
    read_cols: set[str] = set()
    target_cnt_col = None
    total_col = config.TOTAL_GENES_COL if (config.TOTAL_GENES_COL in dset.schema.names) else None
    if biotype_pct_filter and biotype_pct_filter.get("biotype"):
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        candidate = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if candidate in dset.schema.names:
            target_cnt_col = candidate
            read_cols.add(candidate)
        if total_col:
            read_cols.add(total_col)
        else:
            col_map = list_biotype_columns()
            for c in col_map.get("count", []):
                if c in dset.schema.names:
                    read_cols.add(c)
    else:
        # If no pct filter, read a single light column for counting
        first_col = dset.schema.names[0]
        read_cols.add(first_col)

    scanner = ds.Scanner.from_dataset(
        dset,
        columns=list(read_cols),
        filter=expr,
        batch_size=8192,
    )

    total = 0
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

        if target_cnt_col:
            numer = pd.to_numeric(pdf[target_cnt_col], errors="coerce").astype(float)
            if total_col and total_col in pdf.columns:
                denom = pd.to_numeric(pdf[total_col], errors="coerce").astype(float)
            else:
                col_map = list_biotype_columns()
                cnt_cols = [c for c in col_map.get("count", []) if c in pdf.columns]
                denom = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce") for c in cnt_cols}).sum(axis=1).astype(
                    float)

            valid = denom > 0
            pct = pd.Series(np.nan, index=pdf.index, dtype="float64")
            pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

            pmin = float(biotype_pct_filter.get("min", 0.0))
            pmax = float(biotype_pct_filter.get("max", 100.0))
            mask = pct.ge(pmin) & pct.le(pmax)
            mask = mask.fillna(False)
            total += int(mask.sum())
        else:
            total += int(len(pdf))

    return total


def _coerce_values_for_column(dset: ds.Dataset, column: str, vals: Sequence) -> list:
    """Coerce dropdown string values to the Arrow column dtype (int/float/str)."""
    if not vals:
        return []
    try:
        pa_type = dset.schema.field(column).type
    except Exception:
        pa_type = pa.string()

    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        try:
            if pat.is_integer(pa_type):
                out.append(int(float(s)))
            elif pat.is_floating(pa_type):
                out.append(float(s))
            else:
                out.append(s)
        except Exception:
            # skip values that can't be coerced
            continue
    return out


# For taxonomy filters cascade
def distinct_values_for_column(
    column: str,
    *,
    taxonomy_filter_map: dict[str, Sequence[str]] | None  = None,
    taxonomy_filter: Sequence[str] | None = None,   # legacy single-col (optional)
    climate_filter: Sequence[str] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    biotype_pct_filter: dict | None = None,
    climate_ranges: dict | None = None,
    biogeo_ranges: dict | None = None,
    limit: int = 5000,
) -> list[str]:
    """
    Return sorted distinct values for `column` from dashboard_main parquet,
    under the SAME predicate pushdown as the grid (type-aware),
    and honoring an optional biotype-% row filter.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # biogeo → accession set (if any)
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return []
        accession_filter = list(accs)

    # Base predicate (taxonomy + climate + accession)
    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
    )

    # Add numeric ranges for clim and biogeo
    r1 = _build_range_expr(dset, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(dset, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    # Try percentage pushdown first
    pct_expr, _pct_col = _build_biotype_pct_pushdown(dset, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)
        # Easy path: pushdown exists → project single column and unique
        table = dset.to_table(columns=[column], filter=expr)
        ser = pd.Series(table.column(0).to_pandas())
        vals = (
            ser.dropna()
               .astype(str)
               .unique()
               .tolist()
        )
        return sorted(vals)[:limit]

    # No pushdown: we may need to compute a row-mask from counts/denominator
    # Read the rank column + whatever is needed to build the mask
    read_cols: set[str] = {column}
    target_cnt_col = None
    total_col = config.TOTAL_GENES_COL if (config.TOTAL_GENES_COL in dset.schema.names) else None
    pmin = pmax = None

    if biotype_pct_filter and biotype_pct_filter.get("biotype"):
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        candidate = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if candidate in dset.schema.names:
            target_cnt_col = candidate
            pmin = float(biotype_pct_filter.get("min", 0.0))
            pmax = float(biotype_pct_filter.get("max", 100.0))
            read_cols.add(candidate)
            if total_col:
                read_cols.add(total_col)
            else:
                # need all *_count columns to compute denom
                col_map = list_biotype_columns()
                for c in col_map.get("count", []):
                    if c in dset.schema.names:
                        read_cols.add(c)

    # If no biotype filter active, we can just project the column
    if not target_cnt_col:
        table = dset.to_table(columns=[column], filter=expr)
        ser = pd.Series(table.column(0).to_pandas())
        vals = ser.dropna().astype(str).unique().tolist()
        return sorted(vals)[:limit]

    # Build scanner with minimal columns to compute mask
    scanner = ds.Scanner.from_dataset(
        dset, columns=list(read_cols), filter=expr, batch_size=8192
    )
    uniques: set[str] = set()
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
        pct = pd.Series(np.nan, index=pdf.index, dtype="float64")
        pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

        mask = pct.ge(pmin).where(~pct.isna(), False) & pct.le(pmax).where(~pct.isna(), False)
        sub = pdf.loc[mask, column].dropna().astype(str).unique().tolist()
        uniques.update(sub)

        if len(uniques) >= limit:
            break

    return sorted(list(uniques))[:limit]



def summarize_biotypes_by_rank(
    *,
    group_rank: str,
    biotype_cols: list[str] | None,     # concrete *_count column names or None -> auto-detect
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
    taxonomy_filter: Sequence[str] | None = None,   # legacy
    climate_filter: Sequence[str] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    biotype_pct_filter: dict | None = None,
    climate_ranges: dict | None = None,
    biogeo_ranges: dict | None = None,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Return tidy DF ['group','biotype','value'] where `value` is the **percentage**
    of each biotype within each `group_rank`, computed from *_count columns.
    If TOTAL_GENES_COL exists, percentages are computed as:
        sum(biotype_count) / sum(total_gene_biotypes) * 100
    else:
        sum(biotype_count) / sum(all_biotype_counts_included) * 100
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)
    if not dset or group_rank not in dset.schema.names:
        return pd.DataFrame(columns=["group", "biotype", "value"])

    # Resolve accession filter from biogeo
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return pd.DataFrame(columns=["group", "biotype", "value"])
        accession_filter = list(accs)

    # Predicate (type-aware)
    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
    )

    r1 = _build_range_expr(dset, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(dset, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    #  --- NEW: try to push down the % filter before scanning ---
    pct_expr, _ = _build_biotype_pct_pushdown(dset, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    # Determine which *_count columns to read
    if not biotype_cols:
        col_map = list_biotype_columns()
        cols = list(col_map.get("count", []))
    else:
        cols = list(biotype_cols)
    if not cols:
        return pd.DataFrame(columns=["group", "biotype", "value"])

    # Optional total column
    total_col = None
    if config.TOTAL_GENES_COL and config.TOTAL_GENES_COL in dset.schema.names:
        total_col = config.TOTAL_GENES_COL

    # ---- biotype-% row filter setup (optional) ----
    pct_use = None
    pct_min = pct_max = None
    extra_filter_col = None
    if biotype_pct_filter and biotype_pct_filter.get("biotype") and pct_expr is None:
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        target = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if target in dset.schema.names:
            pct_use = target
            pct_min = float(biotype_pct_filter.get("min", 0.0))
            pct_max = float(biotype_pct_filter.get("max", 100.0))
            # ensure we read the target count column even if not in 'cols'
            if target not in cols:
                extra_filter_col = target

    # Columns to read
    read_cols = [group_rank] + cols
    if total_col: read_cols.append(total_col)
    if extra_filter_col: read_cols.append(extra_filter_col)

    # Accumulators
    sums: dict[tuple[str, str], float] = {}
    group_totals: dict[str, float] = {}

    scanner = ds.Scanner.from_dataset(dset, columns=read_cols, filter=expr, batch_size=batch_size)
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

        # ---- apply biotype-% row filter if requested ----
        if pct_use:
            # Choose denominator: total_gene_biotypes if present; else sum of *_count columns in this batch row
            if total_col and total_col in pdf.columns:
                denom = pd.to_numeric(pdf[total_col], errors="coerce").astype(float)
            else:
                denom = pd.DataFrame({c: pd.to_numeric(pdf.get(c), errors="coerce") for c in cols}).sum(axis=1).astype(
                    float)

            numer = pd.to_numeric(pdf[pct_use], errors="coerce").astype(float)

            # Robust percentage: only divide where denom > 0; leave NaN otherwise (no inf)
            valid = denom > 0
            pct = pd.Series(np.nan, index=pdf.index, dtype="float64")
            pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

            # Range mask; NaNs become False
            mask = pct.ge(pct_min) & pct.le(pct_max)
            mask = mask.fillna(False)

            pdf = pdf[mask]
            if pdf.empty:
                continue

        g = pdf[group_rank].astype(str).fillna("NA")

        # total genes per group (if column present)
        if total_col and total_col in pdf.columns:
            t = pd.to_numeric(pdf[total_col], errors="coerce").fillna(0)
            t_sum = t.groupby(g).sum()
            for grp, val in t_sum.items():
                group_totals[grp] = group_totals.get(grp, 0.0) + float(val)

        # sum each biotype count per group
        for c in cols:
            s = pd.to_numeric(pdf[c], errors="coerce").fillna(0)
            grp_sum = s.groupby(g).sum()
            for grp, val in grp_sum.items():
                key = (grp, c)
                sums[key] = sums.get(key, 0.0) + float(val)

    # Build tidy rows & compute percentages
    cnt_sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
    rows = []

    # precompute fallback denominators = sum of included biotype counts per group
    fallback_totals: dict[str, float] = {}
    for (grp, col), val in sums.items():
        fallback_totals[grp] = fallback_totals.get(grp, 0.0) + val

    groups = sorted({grp for (grp, _) in sums.keys()})
    for grp in groups:
        denom = group_totals.get(grp, 0.0) if total_col else fallback_totals.get(grp, 0.0)
        if denom <= 0:
            continue
        for c in cols:
            key = (grp, c)
            if key not in sums:
                continue
            name = c[:-len(cnt_sfx)] if c.endswith(cnt_sfx) else c
            pct = (sums[key] / denom) * 100.0
            rows.append({"group": grp, "biotype": name, "value": float(pct)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["group"] = df["group"].astype(str)
    # Keep deterministic order
    return df.sort_values(["group", "biotype"]).reset_index(drop=True)


def summarize_biotype_totals(
    *,
    biotype_cols: list[str] | None = None,          # *_count columns or None → auto-detect
    taxonomy_filter_map: dict[str, Sequence[str]] | None = None,
    taxonomy_filter: Sequence[str] | None = None,   # legacy
    climate_filter: Sequence[str] | None = None,
    bio_levels_filter: Sequence[str] | None = None,
    bio_values_filter: Sequence[str] | None = None,
    biotype_pct_filter: dict | None = None,      # NEW: {"biotype": str, "min": float, "max": float}
    climate_ranges: dict | None = None,
    biogeo_ranges: dict | None = None,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Sum *_count columns across ALL matching rows (no grouping), honoring the biotype-% row filter if provided.
    Returns: DataFrame ['biotype','count'] sorted by descending count.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve accession filter from biogeo
    accession_filter = None
    if bio_levels_filter or bio_values_filter:
        accs = _get_accessions_for_biogeo(bio_levels_filter or [], bio_values_filter or [])
        if not accs:
            return pd.DataFrame(columns=["biotype", "count"])
        accession_filter = list(accs)

    # Predicate (type-aware) for taxonomy + accession (+ climate later)
    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
    )

    # Add numeric ranges for clim and biogeo
    r1 = _build_range_expr(dset, climate_ranges)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = _build_range_expr(dset, biogeo_ranges)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    # Which *_count columns to sum
    if not biotype_cols:
        col_map = list_biotype_columns()
        cols = list(col_map.get("count", []))
    else:
        cols = [c for c in biotype_cols if c in dset.schema.names]
    if not cols:
        return pd.DataFrame(columns=["biotype", "count"])

    # Try pushdown for percentage on the chosen biotype (if provided)
    pct_expr, _pct_col = _build_biotype_pct_pushdown(dset, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    # If no pushdown, we may need to compute a row-level mask
    target_cnt_col = None
    total_col = config.TOTAL_GENES_COL if (config.TOTAL_GENES_COL in dset.schema.names) else None
    extra_filter_col = None
    pmin = pmax = None
    if biotype_pct_filter and biotype_pct_filter.get("biotype") and pct_expr is None:
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
        pref = config.GENE_BIOTYPE_PREFIX or ""
        candidate = f"{pref}{biotype_pct_filter['biotype']}{sfx}"
        if candidate in dset.schema.names:
            target_cnt_col = candidate
            pmin = float(biotype_pct_filter.get("min", 0.0))
            pmax = float(biotype_pct_filter.get("max", 100.0))
            if candidate not in cols:
                extra_filter_col = candidate  # ensure we read it to compute mask

    # Columns to read: all *_count we’re summing + any denominator/extra needed to build mask
    read_cols: set[str] = set(cols)
    if extra_filter_col:
        read_cols.add(extra_filter_col)
    if total_col:
        read_cols.add(total_col)
    elif target_cnt_col:
        # When no total column, we’ll need all *_count columns to build the denominator for the mask
        col_map = list_biotype_columns()
        for c in col_map.get("count", []):
            if c in dset.schema.names:
                read_cols.add(c)

    sums: dict[str, float] = {c: 0.0 for c in cols}

    scanner = ds.Scanner.from_dataset(
        dset,
        columns=list(read_cols),
        filter=expr,
        batch_size=batch_size,
    )

    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        pdf = rb.to_pandas(types_mapper=pd.ArrowDtype)

        # If we need a row-level mask for biotype %, compute it now
        if target_cnt_col:
            numer = pd.to_numeric(pdf[target_cnt_col], errors="coerce").astype(float)
            if total_col and total_col in pdf.columns:
                denom = pd.to_numeric(pdf[total_col], errors="coerce").astype(float)
            else:
                # sum of all *_count columns present in this batch
                col_map = list_biotype_columns()
                cnt_cols = [c for c in col_map.get("count", []) if c in pdf.columns]
                denom = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce") for c in cnt_cols}).sum(axis=1).astype(float)

            valid = denom > 0
            pct = pd.Series(np.nan, index=pdf.index, dtype="float64")
            pct.loc[valid] = (numer.loc[valid] / denom.loc[valid]) * 100.0

            mask = pct.ge(pmin).where(~pct.isna(), False) & pct.le(pmax).where(~pct.isna(), False)
            pdf = pdf[mask]
            if pdf.empty:
                continue

        # Sum each requested *_count column over the (optionally) masked rows
        for c in cols:
            s = pd.to_numeric(pdf[c], errors="coerce").fillna(0)
            sums[c] += float(s.sum())

    # Build tidy result (strip "_count")
    cnt_sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
    rows = []
    for c, v in sums.items():
        name = c[:-len(cnt_sfx)] if c.endswith(cnt_sfx) else c
        rows.append({"biotype": name, "count": float(v)})

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=32)
def _get_column_min_max_cached(cols_key: tuple[str, ...]) -> dict[str, tuple[float | None, float | None]]:
    """
    Internal: scan the Parquet once for the requested columns and return {col: (min, max)}.
    Cached by the exact tuple of column names.
    """
    import pyarrow.compute as pc

    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = ds.dataset(main_path)

    result: dict[str, tuple[float, float]] = {}
    for col in cols_key:
        if col not in dset.schema.names:
            result[col] = (None, None)  # caller will fallback
            continue

        gmin: float | None = None
        gmax: float | None = None

        scanner = ds.Scanner.from_dataset(dset, columns=[col], batch_size=65536)
        for batch in scanner.to_batches():
            arr = batch.column(0)
            mm = pc.min_max(arr).as_py()  # {'min': x, 'max': y} or Nones
            bmin, bmax = mm.get("min"), mm.get("max")
            if bmin is None or bmax is None:
                continue
            vmin = float(bmin)
            vmax = float(bmax)
            gmin = vmin if gmin is None else min(gmin, vmin)
            gmax = vmax if gmax is None else max(gmax, vmax)

        result[col] = (gmin, gmax)

    return result


def get_column_min_max(columns: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """
    Public helper: return {col: (min, max)} for the requested numeric columns.
    Falls back to (None, None) per column if the dataset/column is missing.
    Results are cached per unique set of columns.
    """
    # normalize to a stable cache key (order matters only for caching)
    key = tuple(columns)
    return _get_column_min_max_cached(key)

# Explicit public API for parquet I/O helpers (wildcard imports, docs, etc.)
__all__ = [
    "list_columns",
    "pick_default_columns",
    "get_biogeo_tags_for_accessions_by_level",
    "list_biotype_columns",
    "summarize_biotypes",
    "load_dashboard_page",
    "count_dashboard_rows",
    "distinct_values_for_column",
    "summarize_biotypes_by_rank",
    "summarize_biotype_totals",
    "get_column_min_max",
]
