from __future__ import annotations

from typing import Iterable, List, Dict, Sequence
from pathlib import Path
import pandas as pd

from utils import config
from utils.parquet_io import distinct_values_for_column

import pyarrow.dataset as ds

def resolve_preset_columns(all_columns: List[str], patterns: Iterable[str]) -> List[str]:
    pats = list(patterns or [])
    resolved: List[str] = []
    seen = set()

    def matches(col: str, pat: str) -> bool:
        return col.startswith(pat[:-1]) if pat.endswith("*") else (col == pat)

    for col in all_columns:
        for pat in pats:
            if matches(col, pat) and col not in seen:
                seen.add(col)
                resolved.append(col)
                break
    return resolved

def _dataset(path):
    if not path.exists():
        return None
    return ds.dataset(str(path))

def get_filter_options() -> Dict[str, List[Dict[str, str]]]:
    """taxonomy + climate from main parquet; empty lists if missing."""
    out: Dict[str, List[Dict[str, str]]] = {"taxonomy": [], "climate": []}
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not main:
        return out

    # taxonomy
    if config.TAXONOMY_COL in main.schema.names:
        t = main.to_table(columns=[config.TAXONOMY_COL])
        vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
        out["taxonomy"] = [{"label": v, "value": v} for v in sorted(vals)[:5000]]

    # climate
    if config.CLIMATE_LABEL_COL in main.schema.names:
        t = main.to_table(columns=[config.CLIMATE_LABEL_COL])
        vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
        out["climate"] = [{"label": v, "value": v} for v in sorted(vals)[:5000]]

    return out

def list_biogeo_levels(limit: int = 200) -> List[Dict[str, str]]:
    dset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    if not dset or config.BIOGEO_LEVEL_COL not in dset.schema.names:
        return []
    t = dset.to_table(columns=[config.BIOGEO_LEVEL_COL])
    vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    return [{"label": v, "value": v} for v in sorted(vals)[:limit]]

def list_biogeo_values(levels: List[str] | None, limit: int = 5000) -> List[Dict[str, str]]:
    dset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    if not dset or config.BIOGEO_VALUE_COL not in dset.schema.names:
        return []
    expr = None
    if levels:
        expr = ds.field(config.BIOGEO_LEVEL_COL).isin([str(x) for x in levels])
    t = dset.to_table(columns=[config.BIOGEO_VALUE_COL], filter=expr)
    vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    return [{"label": v, "value": v} for v in sorted(vals)[:limit]]

def _dataset(path: Path):
    if not path.exists():
        return None
    return ds.dataset(str(path))

def list_unique_values_for_column(column: str, limit: int = 5000) -> List[dict]:
    dset = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not dset or column not in dset.schema.names:
        return []
    tbl = dset.to_table(columns=[column])
    vals = pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    vals = sorted(vals)[:limit]
    return [{"label": v, "value": v} for v in vals]

def list_taxonomy_options() -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for col in (config.TAXONOMY_RANK_COLUMNS or []):
        out[col] = list_unique_values_for_column(col)
    # tax_id can be large; include if present
    if "tax_id" in (config.TAXONOMY_RANK_COLUMNS or []) or True:
        out["tax_id"] = list_unique_values_for_column("tax_id")
    return out

#For taxonomy filter cascading
def _dataset(path):
    if not path.exists():
        return None
    return ds.dataset(str(path))


def _coerce_for_column(dset: ds.Dataset, column: str, vals: Sequence) -> list:
    """Coerce dropdown strings to the Arrow dtype of `column` (handles tax_id ints)."""
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
            continue
    return out


# For cascading taxonomy filters
def list_rank_values_with_filters(
    rank: str,
    higher_rank_selections: Dict[str, Sequence[str]] | None,
    limit: int = 5000,
) -> List[dict]:
    """
    Return distinct values for `rank`, filtered by the selections in higher ranks.
    Example: rank='genus', higher_rank_selections={'family': ['Rosaceae']}
    """
    dset = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not dset or rank not in dset.schema.names:
        return []

    # Build AND filter from higher ranks
    expr = None
    for col, vals in (higher_rank_selections or {}).items():
        if not vals or col not in dset.schema.names:
            continue
        coerced = _coerce_for_column(dset, col, vals)
        if not coerced:
            # Selection couldn't be coerced → no matches; return empty list now
            return []
        e = ds.field(col).isin(coerced)
        expr = e if expr is None else (expr & e)

    # Project only the target rank column, apply filter
    table = dset.to_table(columns=[rank], filter=expr)
    vals = (
        pd.Series(table.column(0).to_pandas())
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    vals = sorted(vals)[:limit]
    return [{"label": v, "value": v} for v in vals]


def list_taxonomy_options_cascaded(
    selections: Dict[str, Sequence[str]] | None,
) -> Dict[str, List[dict]]:
    """
    For each rank in TAXONOMY_RANK_COLUMNS, list options filtered by all *higher* ranks.
    Within a rank it's OR; across ranks it's AND (same as the grid).
    """
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    sels = selections or {}
    out: Dict[str, List[dict]] = {}

    # Left→right: build options for rank_i given ranks[:i] selections
    for i, rank in enumerate(ranks):
        higher_map = {r: sels.get(r, []) for r in ranks[:i] if sels.get(r)}
        values = distinct_values_for_column(rank, taxonomy_filter_map=higher_map)
        out[rank] = [{"label": v, "value": v} for v in values]

    # tax_id: filter by *all* rank selections (if present)
    higher_all = {r: sels.get(r, []) for r in ranks if sels.get(r)}
    taxid_values = distinct_values_for_column("tax_id", taxonomy_filter_map=higher_all)
    out["tax_id"] = [{"label": str(v), "value": str(v)} for v in taxid_values]
    return out
