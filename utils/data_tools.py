from __future__ import annotations

from typing import Iterable, List, Dict
import pandas as pd

from utils import config
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
