# utils/data_tools.py
# -----------------------------------------------------------------------------
# Utilities for packing/pruning the global filters store and small UI helpers.
# Keep pure functions here (no side effects); callbacks should stay thin and
# delegate to these helpers. See utils/types.GlobalFilters for the contract.
#
# Sections:
# - Common option builders (taxonomy, climate, biogeo)
# - Taxonomy cascade helpers
# - Home KPIs helpers
# - Data Browser helpers
# - Biotype vs Environment helpers
# - Genome Annotations helpers
# -----------------------------------------------------------------------------
import math
from functools import lru_cache
from typing import Iterable, Mapping, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from utils import config
from utils.parquet_io import (
    distinct_values_for_column,
    list_biotype_columns,
    _dataset as _io_dataset,
    _build_filter_expr,
    _build_range_expr,
    _build_biotype_pct_pushdown,
)
from utils.types import (
    GlobalFilters,
    TaxRank,
    TaxonomyMap,
    ClimateRanges,
    BiogeoRanges,
    BiotypePctFilter,
)
import utils.parquet_io as parquet_io


# ---------------------------------------------------------------------------
# Internal: dataset loader
# ---------------------------------------------------------------------------

def _dataset(path: Path | str) -> ds.Dataset | None:
    p = Path(path)
    if not p.exists():
        return None
    return ds.dataset(str(p))


# ---------------------------------------------------------------------------
# Common option builders (taxonomy, climate, biogeo)
# ---------------------------------------------------------------------------

def resolve_preset_columns(
        all_columns: list[str],
        patterns: Iterable[str]
) -> list[str]:
    """
    Given a list of columns and simple wildcard patterns (suffix '*'),
    return columns in the order they appear in `all_columns`.
    """
    pats = list(patterns or [])
    resolved: list[str] = []
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


def list_unique_values_for_column(column: str, limit: int = 5000) -> list[dict]:
    dset = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not dset or column not in dset.schema.names:
        return []
    tbl = dset.to_table(columns=[column])
    vals = pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    vals = sorted(vals)[:limit]
    return [{"label": v, "value": v} for v in vals]


@lru_cache(maxsize=1)
def list_taxonomy_options() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for col in (config.TAXONOMY_RANK_COLUMNS or []):
        out[col] = list_unique_values_for_column(col)
    out["tax_id"] = list_unique_values_for_column("tax_id")
    return out


def get_filter_options() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {"taxonomy": [], "climate": []}
    main = _dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not main:
        return out

    if config.TAXONOMY_COL and config.TAXONOMY_COL in main.schema.names:
        t = main.to_table(columns=[config.TAXONOMY_COL])
        vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
        out["taxonomy"] = [{"label": v, "value": v} for v in sorted(vals)[:5000]]

    if config.CLIMATE_LABEL_COL in main.schema.names:
        t = main.to_table(columns=[config.CLIMATE_LABEL_COL])
        vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
        out["climate"] = [{"label": v, "value": v} for v in sorted(vals)[:5000]]

    return out


def list_biogeo_levels(limit: int = 200) -> list[dict[str, str]]:
    dset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    if not dset or config.BIOGEO_LEVEL_COL not in dset.schema.names:
        return []
    t = dset.to_table(columns=[config.BIOGEO_LEVEL_COL])
    vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    return [{"label": v, "value": v} for v in sorted(vals)[:limit]]


def list_biogeo_values(
        levels: list[str] | None,
        limit: int = 5000
) -> list[dict[str, str]]:
    dset = _dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    if not dset or config.BIOGEO_VALUE_COL not in dset.schema.names:
        return []
    expr = None
    if levels:
        expr = ds.field(config.BIOGEO_LEVEL_COL).isin([str(x) for x in levels])
    t = dset.to_table(columns=[config.BIOGEO_VALUE_COL], filter=expr)
    vals = pd.Series(t.column(0).to_pandas()).dropna().astype(str).unique().tolist()
    return [{"label": v, "value": v} for v in sorted(vals)[:limit]]


# ---------------------------------------------------------------------------
# Taxonomy cascade helpers (used by global filters)
# ---------------------------------------------------------------------------

def list_rank_values_with_filters(
    rank: str,
    higher_rank_selections: dict[str, Sequence[str]] | None,
    limit: int = 5000,
) -> list[dict]:
    ranks_map = higher_rank_selections or {}
    values = distinct_values_for_column(rank, taxonomy_filter_map=ranks_map)
    values = sorted(values)[:limit]
    return [{"label": v, "value": v} for v in values]


def list_taxonomy_options_cascaded(
        selections: dict[str, Sequence[str]] | None
) -> dict[str, list[dict]]:
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    sels = selections or {}
    out: dict[str, list[dict]] = {}

    for i, rank in enumerate(ranks):
        higher_map = {r: sels.get(r, []) for r in ranks[:i] if sels.get(r)}
        values = distinct_values_for_column(rank, taxonomy_filter_map=higher_map)
        out[rank] = [{"label": v, "value": v} for v in values]

    higher_all = {r: sels.get(r, []) for r in ranks if sels.get(r)}
    taxid_values = distinct_values_for_column("tax_id", taxonomy_filter_map=higher_all)
    out["tax_id"] = [{"label": str(v), "value": str(v)} for v in taxid_values]
    return out


# ---------------------------------------------------------------------------
# Global Filters helpers (used by callbacks/global_filters.py)
# ---------------------------------------------------------------------------

def gf_clean_list(v: Sequence[str | None]) -> list[str]:
    """
    Normalize a user-supplied list by:
      - removing None/empty/whitespace-only entries
      - stripping whitespace from each remaining string
    Returns a new list; input is not mutated.
    """
    if v is None:
        return []
    if isinstance(v, (str, int, float)):
        v = [v]
    try:
        out = [str(x) for x in v if x not in (None, "", [])]
    except Exception:
        return []
    # preserve order, dedupe
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def gf_is_full_span(lo: float, hi: float, span_lo: float, span_hi: float, tol: float = 0.0) -> bool:
    """
    Strict full-span check.
    Returns True iff [lo, hi] equals [span_lo, span_hi] within absolute tolerance `tol`.
    Any narrowing (lo > span_lo or hi < span_hi) returns False.

    Examples (tol=0):
      (0, 100)  vs (0, 100)  -> True
      (0, 99.9) vs (0, 100)  -> False
      (10, 90)  vs (0, 100)  -> False
    """
    try:
        lo_f, hi_f = float(lo), float(hi)
        s_lo, s_hi = float(span_lo), float(span_hi)
    except Exception:
        return False
    return (abs(lo_f - s_lo) <= tol) and (abs(hi_f - s_hi) <= tol)


def gf_build_climate_ranges(
        b1_val: Sequence[float] | None,
        b1_min: float | None,
        b1_max: float | None,
        b12_val: Sequence[float] | None,
        b12_min: float | None,
        b12_max: float | None
) -> ClimateRanges:
    """
    Return only narrowed climate ranges as {col: (lo, hi)}.
    Full-span means “no filter” (omit the key).
    """
    out: ClimateRanges = {}

    # Annual Mean Temperature
    try:
        b1_lo, b1_hi = float(b1_val[0]), float(b1_val[1])
        if not gf_is_full_span(b1_lo, b1_hi, float(b1_min), float(b1_max)):
            out["clim_bio1_mean"] = (b1_lo, b1_hi)
    except Exception:
        pass  # ignore malformed inputs

    # Annual Precipitation
    try:
        b12_lo, b12_hi = float(b12_val[0]), float(b12_val[1])
        if not gf_is_full_span(b12_lo, b12_hi, float(b12_min), float(b12_max)):
            out["clim_bio12_mean"] = (b12_lo, b12_hi)
    except Exception:
        pass

    return out


def gf_build_biogeo_ranges(
        range_val: Sequence[float] | None,
        rmin: float | None,
        rmax: float | None
) -> BiogeoRanges:
    """
    Return only narrowed biogeographic numeric ranges as {col: (lo, hi)}.
    Full-span means “no filter” (omit the key).
    """
    out: BiogeoRanges = {}
    try:
        lo, hi = float(range_val[0]), float(range_val[1])
        if not gf_is_full_span(lo, hi, float(rmin), float(rmax)):
            out["range_km2"] = (lo, hi)
    except Exception:
        pass
    return out

def gf_build_taxonomy_map_from_values(
        ranks: Sequence[TaxRank] | Sequence[str],
        values_by_rank: Mapping[TaxRank, Sequence[str]] | Mapping[str, Sequence[str]]
) -> TaxonomyMap:
    """
    Build a compact TaxonomyMap from a rank->list mapping by:
      - dropping ranks whose list is empty
      - preserving the order of incoming lists
    Returns a dict keyed by TaxRank with non-empty string lists.
    """
    tmap = {}
    for r in ranks:
        vals = gf_clean_list(values_by_rank.get(r))
        if vals:
            tmap[r] = vals
    return tmap

def gf_build_biotype_pct(
        biotype: str | None,
        pct_range: Sequence[float] | None
) -> BiotypePctFilter:
    """Return {'biotype','min','max'} only when range is narrowed; else None."""
    if not biotype:
        return None
    try:
        lo, hi = float(pct_range[0]), float(pct_range[1])
    except Exception:
        return None
    # Full span (0..100) means “no filter” -> omit from store
    if lo <= 0.0 and hi >= 100.0:
        return None
    return {"biotype": str(biotype), "min": lo, "max": hi}


def gf_build_store(
    taxonomy_map: TaxonomyMap | None = None,
    climate_labels: Sequence[str] | None = None,
    bio_levels: Sequence[str] | None = None,
    bio_values: Sequence[str] | None = None,
    climate_ranges: ClimateRanges | None = None,
    biogeo_ranges: BiogeoRanges | None = None,
    biotype_pct: BiotypePctFilter | None = None,
) -> GlobalFilters:
    """
    Pack the global filters store (GlobalFilters). Only include keys that are
    non-empty / narrowed, per the store contract. Inputs may be None/empty;
    outputs never contain empty keys.
    """
    store: GlobalFilters = {}
    if taxonomy_map:
        store["taxonomy_map"] = taxonomy_map
    if climate_labels:
        store["climate"] = gf_clean_list(climate_labels)
    if bio_levels:
        store["bio_levels"] = gf_clean_list(bio_levels)
    if bio_values:
        store["bio_values"] = gf_clean_list(bio_values)
    if climate_ranges:
        store["climate_ranges"] = climate_ranges
    if biogeo_ranges:
        store["biogeo_ranges"] = biogeo_ranges
    if biotype_pct and biotype_pct.get("biotype"):
        store["biotype_pct"] = biotype_pct
    return store


def gf_build_quartile_int_marks(vmin: float, vmax: float) -> dict[int, str]:
    """
    Build sparse slider marks at ~quartiles for a numeric range:
    min, 25%, 50%, 75%, max.

    - Keys are integers so Dash reliably renders labels.
    - Values are strings (e.g., "1000").
    - Sliders remain continuous; these are labels only.

    Returns empty dict when inputs are invalid.
    """
    try:
        lo = float(vmin)
        hi = float(vmax)
    except Exception:
        return {}

    if not (math.isfinite(lo) and math.isfinite(hi)):
        return {}

    if hi < lo:
        lo, hi = hi, lo

    if hi == lo:
        v = int(round(lo))
        return {v: str(v)}

    span = hi - lo
    ticks = [lo, lo + 0.25 * span, lo + 0.50 * span, lo + 0.75 * span, hi]
    ints = [int(round(t)) for t in ticks]

    # Deduplicate in case rounding collapses adjacent values
    marks: dict[int, str] = {}
    for t in ints:
        if t not in marks:
            marks[t] = str(t)
    return marks

# ---------------------------------------------------------------------------
# Home KPIs helpers (used by callbacks/home_kpis.py)
# ---------------------------------------------------------------------------

def kpi_format_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def kpi_filtered_accessions(
    taxonomy_filter_map: dict[str, Sequence[str]] | None,
    climate_filter: Sequence[str] | None,
    bio_levels_filter: Sequence[str] | None,
    bio_values_filter: Sequence[str] | None,
    biotype_pct_filter: dict | None,
    climate_ranges=None,
    biogeo_ranges=None,
) -> set[str]:
    main = _io_dataset(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    if not main or not config.ACCESSION_COL_MAIN:
        return set()

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

    if bio_levels_filter or bio_values_filter:
        bset = _io_dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
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

    pct_expr, _ = _build_biotype_pct_pushdown(main, biotype_pct_filter)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    tbl = main.to_table(columns=[config.ACCESSION_COL_MAIN], filter=expr)
    return set(pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique().tolist())


def kpi_biogeo_distinct_counts(accessions: set[str] | list[str]) -> tuple[int, int, int]:
    if not accessions:
        return (0, 0, 0)
    bset = _io_dataset(config.DATA_DIR / config.BIOGEO_LONG_FN)
    if not bset:
        return (0, 0, 0)

    acc_field = ds.field(config.ACCESSION_COL_BIOGEO).isin(list(accessions))

    def _count(level_name: str) -> int:
        filt = acc_field & (ds.field(config.BIOGEO_LEVEL_COL) == level_name)
        tbl = bset.to_table(columns=[config.BIOGEO_VALUE_COL], filter=filt)
        vals = pd.Series(tbl.column(0).to_pandas()).dropna().astype(str).unique()
        return int(len(vals))

    return _count("realm"), _count("biome"), _count("ecoregion")


# ---------------------------------------------------------------------------
# Data Browser helpers (used by callbacks/data_browser_callbacks.py)
# ---------------------------------------------------------------------------

def db_to_markdown_link(v: str) -> str:
    """Render a URL (or bare host) as a Markdown link label=hostname."""
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    href = s if s.lower().startswith(("http://", "https://")) else f"https://{s}"
    try:
        from urllib.parse import urlparse
        host = urlparse(href).hostname or href
        label = host.replace("www.", "")
    except Exception:
        label = href
    return f"[{label}]({href})"


def db_make_column_defs(df: pd.DataFrame) -> list[dict]:
    """Build dash-ag-grid columnDefs with URL columns using markdown renderer."""
    url_cols = set(getattr(config, "URL_COLUMNS", []) or [])
    defs: list[dict] = []
    for col in df.columns:
        c = {"headerName": col, "field": col}
        if col in url_cols:
            c.update({
                "cellRenderer": "markdown",
                "filter": "agTextColumnFilter",
                "minWidth": 160,
            })
        elif pd.api.types.is_numeric_dtype(df[col]):
            c.update({"type": "rightAligned", "filter": "agNumberColumnFilter"})
        else:
            c.update({"filter": "agTextColumnFilter"})
        defs.append(c)
    return defs


# ---------------------------------------------------------------------------
# Biotype vs Environment helpers (used by callbacks/biotype_environment_callbacks.py)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def biotype_pct_columns() -> list[str]:
    colmap = list_biotype_columns()
    return list(colmap.get("pct", []))


def pct_to_count(col_pct: str) -> str:
    base = col_pct.removesuffix(config.GENE_BIOTYPE_PCT_SUFFIX)
    return f"{config.GENE_BIOTYPE_PREFIX}{base}{config.GENE_BIOTYPE_COUNT_SUFFIX}"


def sizes_from_total(total: pd.Series) -> np.ndarray:
    t = pd.to_numeric(total, errors="coerce").to_numpy(dtype=float)
    t[np.isinf(t)] = np.nan
    if np.all(~np.isfinite(t)):
        return np.full_like(t, 8.0, dtype=float)
    finite = t[np.isfinite(t)]
    if finite.size < 2:
        return np.full_like(t, 8.0, dtype=float)
    q_lo, q_hi = np.nanquantile(finite, [0.05, 0.95])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        q_lo, q_hi = np.nanmin(finite), np.nanmax(finite)
        if not np.isfinite(q_hi) or q_hi <= q_lo:
            return np.full_like(t, 8.0, dtype=float)
    norm = np.clip((t - q_lo) / max(q_hi - q_lo, 1.0), 0.0, 1.0)
    return 6.0 + 10.0 * np.sqrt(norm)  # 6–16 px


def stable_sample(df: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


def get_accessions_for_biogeo(levels: list[str], values: list[str]) -> list[str]:
    if not levels and not values:
        return []
    acc_set = parquet_io._get_accessions_for_biogeo(levels, values)
    return sorted(list(acc_set))


# ---------------------------------------------------------------------------
# Genome Annotations helpers (used by callbacks/genome_annotations_callbacks.py)
# ---------------------------------------------------------------------------

def ga_next_rank(current: str | None) -> str | None:
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if not current or current not in ranks:
        return None
    i = ranks.index(current)
    return ranks[i + 1] if i + 1 < len(ranks) else None


def ga_prev_rank(current: str | None) -> str | None:
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if not current or current not in ranks:
        return None
    i = ranks.index(current)
    return ranks[i - 1] if i - 1 >= 0 else None


def ga_apply_drill_to_taxonomy_map(
    drill_path: list[dict] | None,
    taxonomy_map: TaxonomyMap | None,
) -> TaxonomyMap:
    """
    Merge a list of drill steps (from bar clicks) into a TaxonomyMap.

    Each step is expected to look like: {"rank": <rank>, "value": <value>}.

    Rules
    -----
    - Only apply steps for known taxonomy ranks (config.TAXONOMY_RANK_COLUMNS, plus 'tax_id').
    - Ignore empty/None/whitespace values.
    - Avoid duplicates; preserve insertion order for values at a given rank.
    - Seed from the incoming taxonomy_map (if any), but only keep known ranks.

    Returns
    -------
    TaxonomyMap: {rank: [values], ...} with non-empty lists.
    """
    # Allowed ranks: configured ranks + 'tax_id' (if used in your UI).
    allowed_ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if "tax_id" not in allowed_ranks:
        allowed_ranks.append("tax_id")

    # Start from the existing map, but only keep allowed ranks and clean values.
    tmap: TaxonomyMap = {}
    src = taxonomy_map or {}
    for r in allowed_ranks:
        vals = src.get(r)
        if vals:
            cleaned = [str(x).strip() for x in vals if x not in (None, "")]
            if cleaned:
                tmap[r] = cleaned

    # Apply drill steps safely.
    for step in (drill_path or []):
        r = step.get("rank")
        v = step.get("value")

        if not r or r not in allowed_ranks:
            continue
        if v is None:
            continue

        v_str = str(v).strip()
        if not v_str:
            continue

        existing = list(tmap.get(r, []))
        if v_str not in existing:
            existing.append(v_str)
            tmap[r] = existing

    return tmap



# Explicit public API (import surfaces used across callbacks/pages)
__all__ = [
    # Global-filters helpers
    "gf_clean_list",
    "gf_is_full_span",
    "gf_build_taxonomy_map_from_values",
    "gf_build_climate_ranges",
    "gf_build_biogeo_ranges",
    "gf_build_biotype_pct",
    "gf_build_store",
    "gf_build_quartile_int_marks",
    # Option list helpers
    "list_taxonomy_options",
    "list_biogeo_levels",
    "list_biogeo_values",
    "list_taxonomy_options_cascaded",
    "biotype_pct_columns",
]