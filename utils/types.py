# utils/types.py
from typing import Literal, TypedDict


# ---- Taxonomy types ---------------------------------------------------------

# The canonical rank keys we pass around in the filters store.
TaxRank = Literal[
    "kingdom", "phylum", "class", "order", "family", "genus", "species", "tax_id"
]

# Map of rank -> selected values (non-empty lists only make it into the store).
TaxonomyMap = dict[TaxRank, list[str]]


# ---- Range types (numeric sliders) -----------------------------------------

# Generic "column -> (lo, hi)" range mapping. We use it for climate + biogeo.
RangeTuple = tuple[float, float]
ClimateRanges = dict[str, RangeTuple]
BiogeoRanges = dict[str, RangeTuple]


# ---- Biotype% filter --------------------------------------------------------

class BiotypePctFilter(TypedDict):
    """
    Filter for a single gene biotype percentage:
      - biotype: base column name without the percentage suffix
      - min/max: inclusive bounds in [0.0, 100.0]
    """
    biotype: str
    min: float
    max: float


# ---- Global Filters store (single source of truth) --------------------------

class GlobalFilters(TypedDict, total=False):
    """
    Typed view of dcc.Store(id="global-filters"). Keys are present only when
    non-empty / narrowed. This mirrors the agreed contract exactly:

    {
      "taxonomy_map": {rank: [values], ...},
      "climate": [labels],
      "bio_levels": [levels],
      "bio_values": [values],
      "climate_ranges": {"clim_bio1_mean":[lo,hi], "clim_bio12_mean":[lo,hi]},
      "biogeo_ranges": {"range_km2":[lo,hi]},
      "biotype_pct": {"biotype":"<base>", "min": float, "max": float}
    }
    """
    taxonomy_map: TaxonomyMap
    climate: list[str]
    bio_levels: list[str]
    bio_values: list[str]
    climate_ranges: ClimateRanges
    biogeo_ranges: BiogeoRanges
    biotype_pct: BiotypePctFilter


__all__ = [
    "TaxRank",
    "TaxonomyMap",
    "RangeTuple",
    "ClimateRanges",
    "BiogeoRanges",
    "BiotypePctFilter",
    "GlobalFilters",
]
