from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.types as pat
import pyarrow.dataset as ds

from utils import config

def _dataset(path: Path) -> ds.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return ds.dataset(str(path))

def list_columns(path: Path) -> List[str]:
    dset = _dataset(path)
    return list(dset.schema.names)

def pick_default_columns(all_cols: Sequence[str], max_cols: int = 25) -> List[str]:
    return list(all_cols[:max_cols])

def _get_accessions_for_biogeo(levels: Sequence[str] | None, values: Sequence[str] | None) -> Set[str]:
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

def get_biogeo_tags_for_accessions_by_level(accessions: Sequence[str], levels: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
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
    out: Dict[str, Dict[str, List[str]]] = {}
    for acc, g_acc in pdf.groupby(config.ACCESSION_COL_BIOGEO):
        slot: Dict[str, List[str]] = {}
        for lvl, g_lvl in g_acc.groupby(config.BIOGEO_LEVEL_COL):
            vals = g_lvl[config.BIOGEO_VALUE_COL].dropna().astype(str).unique().tolist()
            slot[str(lvl)] = sorted(vals)
        out[acc] = slot
    return out

def _build_filter_expr(
    dset: ds.Dataset,
    taxonomy_filter: Optional[Sequence[str]],
    climate_filter: Optional[Sequence[str]],
    accession_filter: Optional[Sequence[str]],
    taxonomy_filter_map: Optional[Dict[str, Sequence[str]]] = None,
):
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
    metric: str,  # "pct" (mean across rows) or "count" (sum across rows)
    biotype_cols: list[str] | None,
    taxonomy_filter: Optional[Sequence[str]],
    climate_filter: Optional[Sequence[str]],
    bio_levels_filter: Optional[Sequence[str]],
    bio_values_filter: Optional[Sequence[str]],
    batch_size: int = 8192,
) -> pd.DataFrame:
    """
    Efficiently summarize biotype columns over ALL matching rows.
      - metric="pct": mean of *_percentage columns (ignoring NaNs)
      - metric="count": sum of *_count columns
    Returns DataFrame with ['biotype','value'].
    """
    assert metric in ("pct", "count")
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve accession filter from biogeo level/value
    levels = bio_levels_filter or []
    values = bio_values_filter or []
    acc_filter = None
    if levels or values:
        accs = _get_accessions_for_biogeo(levels, values)
        if not accs:
            return pd.DataFrame(columns=["biotype", "value"])
        acc_filter = list(accs)

    expr = _build_filter_expr(taxonomy_filter, climate_filter, acc_filter)

    # Work out which columns to read
    available = set(list_columns(main_path))
    if not biotype_cols:
        # default to all matching columns for the chosen metric
        col_map = list_biotype_columns()
        cols = col_map["pct"] if metric == "pct" else col_map["count"]
    else:
        cols = [c for c in biotype_cols if c in available]
    if not cols:
        return pd.DataFrame(columns=["biotype", "value"])

    # Aggregate by streaming
    sums: dict[str, float] = {c: 0.0 for c in cols}
    counts: dict[str, int] = {c: 0 for c in cols}  # for pct means

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
    rows = []
    pct_sfx = config.GENE_BIOTYPE_PCT_SUFFIX or "_percentage"
    cnt_sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
    for c in cols:
        # derive biotype display name by stripping suffix
        if metric == "pct" and c.endswith(pct_sfx):
            name = c[: -len(pct_sfx)]
        elif metric == "count" and c.endswith(cnt_sfx):
            name = c[: -len(cnt_sfx)]
        else:
            name = c

        if metric == "pct":
            denom = max(1, counts[c])
            val = (sums[c] / denom) if denom else 0.0
        else:
            val = sums[c]
        rows.append({"biotype": name, "value": float(val)})

    df = pd.DataFrame(rows).dropna(subset=["value"])
    df = df.sort_values("value", ascending=False, kind="mergesort").reset_index(drop=True)
    return df

def load_dashboard_page(
    *,
    columns: Optional[Sequence[str]],
    page_number: int,
    page_size: int,
    taxonomy_filter: Optional[Sequence[str]] = None,
    taxonomy_filter_map: Optional[Dict[str, Sequence[str]]] = None,
    climate_filter: Optional[Sequence[str]] = None,
    bio_levels_filter: Optional[Sequence[str]] = None,
    bio_values_filter: Optional[Sequence[str]] = None,
    region_filter: Optional[Sequence[str]] = None,  # legacy
    batch_size: int = 8192,
) -> Tuple[pd.DataFrame, int]:
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    all_cols = list_columns(main_path)
    use_cols = list(columns) if columns else pick_default_columns(all_cols, max_cols=25)

    # Resolve biogeo -> accession set
    levels = list(bio_levels_filter) if bio_levels_filter else []
    values = list(bio_values_filter) if bio_values_filter else []
    if region_filter and not values:
        values = list(region_filter)

    accession_filter = None
    if levels or values:
        accs = _get_accessions_for_biogeo(levels, values)
        if not accs:
            return pd.DataFrame(columns=use_cols), 0
        accession_filter = list(accs)

    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
        taxonomy_filter_map=taxonomy_filter_map,
    )

    # Page window
    page = max(1, int(page_number or 1))
    size = max(1, int(page_size or 50))
    offset = (page - 1) * size
    remaining = size

    taken: list[pd.DataFrame] = []
    skipped = 0

    scanner = ds.Scanner.from_dataset(dset, columns=use_cols, filter=expr, batch_size=batch_size)

    for rb in scanner.to_batches():
        n = rb.num_rows
        if n == 0:
            continue
        if skipped + n <= offset:
            skipped += n
            continue

        start_in_batch = max(0, offset - skipped)
        available = n - start_in_batch
        take_now = min(remaining, available)
        if take_now <= 0:
            break

        slice_rb = rb.slice(start_in_batch, take_now)
        taken.append(slice_rb.to_pandas(types_mapper=pd.ArrowDtype))

        remaining -= take_now
        skipped += n
        if remaining <= 0:
            break

    if taken:
        df = pd.concat(taken, ignore_index=True)
    else:
        df = pd.DataFrame(columns=use_cols)
    return df, len(df)

def count_dashboard_rows(
    *,
    taxonomy_filter: Optional[Sequence[str]] = None,
    taxonomy_filter_map: Optional[Dict[str, Sequence[str]]] = None,
    climate_filter: Optional[Sequence[str]] = None,
    bio_levels_filter: Optional[Sequence[str]] = None,
    bio_values_filter: Optional[Sequence[str]] = None,
    region_filter: Optional[Sequence[str]] = None,  # legacy
    batch_size: int = 131072,
) -> int:
    """
    Return the total number of rows in dashboard_main that match the filters.
    Tries Dataset.count_rows(filter=...), and falls back to a light batch scan.
    """
    main_path = config.DATA_DIR / config.DASHBOARD_MAIN_FN
    dset = _dataset(main_path)

    # Resolve biogeo -> accession set
    levels = list(bio_levels_filter) if bio_levels_filter else []
    values = list(bio_values_filter) if bio_values_filter else []
    if region_filter and not values:
        values = list(region_filter)

    accession_filter = None
    if levels or values:
        accs = _get_accessions_for_biogeo(levels, values)
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

    # Fast path (available in recent pyarrow)
    try:
        return int(dset.count_rows(filter=expr))
    except Exception:
        pass

    # Fallback: scan a single lightweight column and sum batch sizes
    cols = list_columns(main_path)
    first_col = config.ACCESSION_COL_MAIN if config.ACCESSION_COL_MAIN in cols else cols[0]
    total = 0
    scanner = ds.Scanner.from_dataset(dset, columns=[first_col], filter=expr, batch_size=batch_size)
    for rb in scanner.to_batches():
        total += rb.num_rows
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
    taxonomy_filter_map: Optional[Dict[str, Sequence[str]]] = None,
    taxonomy_filter: Optional[Sequence[str]] = None,   # legacy single-col (optional)
    climate_filter: Optional[Sequence[str]] = None,
    bio_levels_filter: Optional[Sequence[str]] = None,
    bio_values_filter: Optional[Sequence[str]] = None,
    limit: int = 5000,
) -> list[str]:
    """
    Return sorted distinct values for `column` from dashboard_main parquet,
    under the SAME predicate pushdown as the grid (type-aware).
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

    expr = _build_filter_expr(
        dset=dset,
        taxonomy_filter=taxonomy_filter,
        taxonomy_filter_map=taxonomy_filter_map,
        climate_filter=climate_filter,
        accession_filter=accession_filter,
    )

    # project one column + filter, then unique in pandas (fast enough for options)
    table = dset.to_table(columns=[column], filter=expr)
    ser = pd.Series(table.column(0).to_pandas())
    vals = (
        ser.dropna()
           .astype(str)   # dropdowns use strings
           .unique()
           .tolist()
    )
    return sorted(vals)[:limit]