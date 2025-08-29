# callbacks/global_filters.py
# -----------------------------------------------------------------------------
# Global Filters (single source of truth for cross-page state)
#
# Store contract (keys present only when non-empty/narrowed):
# {
#   "taxonomy_map": {rank: [values], ...},     # ranks = config.TAXONOMY_RANK_COLUMNS + 'tax_id'
#   "climate": [labels],                       # note: param is climate_labels -> store key "climate"
#   "bio_levels": [levels],
#   "bio_values": [values],
#   "climate_ranges": {"clim_bio1_mean": (lo, hi), "clim_bio12_mean": (lo, hi)},
#   "biogeo_ranges": {"range_km2": (lo, hi)},
#   "biotype_pct": {"biotype": "<base>", "min": float, "max": float}
# }
#
# Rules:
# - Full-span sliders = “no filter” ⇒ omit their keys entirely.
# - Sliders’ min/max are data-driven (initialized in navbar).
# - Keep callbacks thin; pack/prune via utils.data_tools.gf_* helpers.
# -----------------------------------------------------------------------------

from dash import Input, Output, State, callback, no_update

from utils import config
from utils.data_tools import (
    # Option builders
    list_taxonomy_options_cascaded,
    list_biogeo_levels,
    list_biogeo_values,
    biotype_pct_columns,

    # Global-filters helpers
    gf_clean_list,
    gf_build_taxonomy_map_from_values,
    gf_build_climate_ranges,
    gf_build_biogeo_ranges,
    gf_build_biotype_pct,
    gf_build_store,
)
from utils.types import (
    GlobalFilters,
    TaxonomyMap,
    ClimateRanges,
    BiogeoRanges,
    BiotypePctFilter
)

# ──────────────────────────────────────────────────────────────────────────────
# A) Biogeography options
# ──────────────────────────────────────────────────────────────────────────────

@callback(
    Output("filter-bio-level", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_biogeo_levels(_):
    # Populate levels once on load/navigation
    return list_biogeo_levels()

@callback(
    Output("filter-bio-value", "options"),
    Output("filter-bio-value", "value"),
    Input("filter-bio-level", "value"),
    State("filter-bio-value", "value"),
    prevent_initial_call=True,
)
def cascade_biogeo_values(levels, cur_values):
    opts = list_biogeo_values(gf_clean_list(levels))
    if not cur_values:
        return opts, []
    valid = {o["value"] for o in (opts or [])}
    pruned = [v for v in (cur_values or []) if v in valid]
    return opts, pruned

# ──────────────────────────────────────────────────────────────────────────────
# B) Taxonomy cascade (options + prune invalid values)
# ──────────────────────────────────────────────────────────────────────────────

@callback(
    # Options (single cascade callback also handles first load)
    Output("filter-tax-kingdom", "options"),
    Output("filter-tax-phylum", "options"),
    Output("filter-tax-class", "options"),
    Output("filter-tax-order", "options"),
    Output("filter-tax-family", "options"),
    Output("filter-tax-genus", "options"),
    Output("filter-tax-species", "options"),
    Output("filter-tax-id", "options"),
    # Values (we prune invalids so UI stays consistent)
    Output("filter-tax-kingdom", "value"),
    Output("filter-tax-phylum", "value"),
    Output("filter-tax-class", "value"),
    Output("filter-tax-order", "value"),
    Output("filter-tax-family", "value"),
    Output("filter-tax-genus", "value"),
    Output("filter-tax-species", "value"),
    Output("filter-tax-id", "value"),
    Input("filter-tax-kingdom", "value"),
    Input("filter-tax-phylum", "value"),
    Input("filter-tax-class", "value"),
    Input("filter-tax-order", "value"),
    Input("filter-tax-family", "value"),
    Input("filter-tax-genus", "value"),
    Input("filter-tax-species", "value"),
    Input("filter-tax-id", "value"),
    prevent_initial_call=False,
)
def cascade_taxonomy(k, p, c, o, f, g, s, taxid):
    selections = {
        "kingdom": gf_clean_list(k),
        "phylum": gf_clean_list(p),
        "class": gf_clean_list(c),
        "order": gf_clean_list(o),
        "family": gf_clean_list(f),
        "genus": gf_clean_list(g),
        "species": gf_clean_list(s),
        "tax_id": gf_clean_list(taxid),
    }
    opts = list_taxonomy_options_cascaded(selections)

    def _prune(cur_vals, options):
        if not cur_vals:
            return []
        valid = {o["value"] for o in (options or [])}
        return [v for v in cur_vals if v in valid]

    k2 = _prune(selections["kingdom"], opts.get("kingdom"))
    p2 = _prune(selections["phylum"],  opts.get("phylum"))
    c2 = _prune(selections["class"],   opts.get("class"))
    o2 = _prune(selections["order"],   opts.get("order"))
    f2 = _prune(selections["family"],  opts.get("family"))
    g2 = _prune(selections["genus"],   opts.get("genus"))
    s2 = _prune(selections["species"], opts.get("species"))
    t2 = _prune(selections["tax_id"],  opts.get("tax_id"))

    return (
        opts.get("kingdom", []), opts.get("phylum", []), opts.get("class", []), opts.get("order", []),
        opts.get("family", []),  opts.get("genus", []),  opts.get("species", []), opts.get("tax_id", []),
        k2, p2, c2, o2, f2, g2, s2, t2
    )

# ──────────────────────────────────────────────────────────────────────────────
# C) Biotype % dropdown (values are base names, label=base)
# ──────────────────────────────────────────────────────────────────────────────

@callback(
    Output("bio-pct-biotype", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_biotype_pct_options(_):
    cols = biotype_pct_columns()  # returns list like ["protein_coding_percentage", ...]
    return [{"label": c.removesuffix(config.GENE_BIOTYPE_PCT_SUFFIX),
             "value": c.removesuffix(config.GENE_BIOTYPE_PCT_SUFFIX)} for c in cols]

# ──────────────────────────────────────────────────────────────────────────────
# D) Sync everything → global store (single source of truth)
#     - Ranges are only included when narrowed (full-span => omitted)
#     - Biotype% stored as {"biotype": <base>, "min":..., "max":...}
# ──────────────────────────────────────────────────────────────────────────────
@callback(
    Output("global-filters", "data", allow_duplicate=True),

    # Taxonomy selections
    Input("filter-tax-kingdom", "value"),
    Input("filter-tax-phylum", "value"),
    Input("filter-tax-class", "value"),
    Input("filter-tax-order", "value"),
    Input("filter-tax-family", "value"),
    Input("filter-tax-genus", "value"),
    Input("filter-tax-species", "value"),
    Input("filter-tax-id", "value"),

    # Climate categorical
    Input("filter-climate", "value"),

    # Biogeo categorical
    Input("filter-bio-level", "value"),
    Input("filter-bio-value", "value"),

    # Climate numeric sliders (+min/max States for full-span detection)
    Input("climate-range-clim_bio1_mean", "value"),
    State("climate-range-clim_bio1_mean", "min"),
    State("climate-range-clim_bio1_mean", "max"),

    Input("climate-range-clim_bio12_mean", "value"),
    State("climate-range-clim_bio12_mean", "min"),
    State("climate-range-clim_bio12_mean", "max"),

    # Distribution numeric slider
    Input("biogeo-range-range_km2", "value"),
    State("biogeo-range-range_km2", "min"),
    State("biogeo-range-range_km2", "max"),

    # Biotype %
    Input("bio-pct-biotype", "value"),
    Input("bio-pct-range", "value"),

    prevent_initial_call="initial_duplicate",
)
def sync_global_store(
    tax_kingdom, tax_phylum, tax_class, tax_order, tax_family, tax_genus, tax_species, tax_id,
    climate_labels,
    bio_levels, bio_values,
    bio1_val, bio1_min, bio1_max,
    bio12_val, bio12_min, bio12_max,
    range_val, rmin, rmax,
    biopct_biotype, biopct_range,
) -> GlobalFilters:
    """Build and return the global filters store.

    Reads all UI controls (taxonomy cascade, climate labels, climate ranges,
    biogeography level/value + range_km2, biotype %) and returns a dict that
    matches utils.types.GlobalFilters.

    Notes
    -----
    - Only include keys when they are non-empty or narrowed.
    - Full-span sliders mean “no filter” and are omitted from the store.
    - The parameter name `climate_labels` is persisted under store key "climate".
    """


    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    values_by_rank = {
        "kingdom": tax_kingdom, "phylum": tax_phylum, "class": tax_class, "order": tax_order,
        "family": tax_family, "genus": tax_genus, "species": tax_species, "tax_id": tax_id,
    }
    taxonomy_map: TaxonomyMap = gf_build_taxonomy_map_from_values(ranks, values_by_rank)
    climate_ranges: ClimateRanges = gf_build_climate_ranges(bio1_val, bio1_min, bio1_max,
                                             bio12_val, bio12_min, bio12_max)
    biogeo_ranges: BiogeoRanges  = gf_build_biogeo_ranges(range_val, rmin, rmax)
    biotype_pct: BiotypePctFilter | None = gf_build_biotype_pct(biopct_biotype, biopct_range)

    store = gf_build_store(
        taxonomy_map=taxonomy_map,
        climate_labels=gf_clean_list(climate_labels),
        bio_levels=gf_clean_list(bio_levels),
        bio_values=gf_clean_list(bio_values),
        climate_ranges=climate_ranges,
        biogeo_ranges=biogeo_ranges,
        biotype_pct=biotype_pct,
    )
    return store

# ──────────────────────────────────────────────────────────────────────────────
# E) Reset buttons — set UI back to neutral; store is recomputed by D)
# ──────────────────────────────────────────────────────────────────────────────

@callback(
    Output("filter-tax-kingdom", "value", allow_duplicate=True),
    Output("filter-tax-phylum", "value", allow_duplicate=True),
    Output("filter-tax-class", "value", allow_duplicate=True),
    Output("filter-tax-order", "value", allow_duplicate=True),
    Output("filter-tax-family", "value", allow_duplicate=True),
    Output("filter-tax-genus", "value", allow_duplicate=True),
    Output("filter-tax-species", "value", allow_duplicate=True),
    Output("filter-tax-id", "value", allow_duplicate=True),
    Input("btn-reset-taxonomy", "n_clicks"),
    prevent_initial_call=True,
)
def reset_taxonomy(n):
    if not n:
        return (no_update,) * 8
    empty = []
    return (empty, empty, empty, empty, empty, empty, empty, empty)

@callback(
    Output("filter-climate", "value", allow_duplicate=True),
    Output("climate-range-clim_bio1_mean", "value", allow_duplicate=True),
    Output("climate-range-clim_bio12_mean", "value", allow_duplicate=True),
    Input("btn-reset-climate", "n_clicks"),
    State("climate-range-clim_bio1_mean", "min"),
    State("climate-range-clim_bio1_mean", "max"),
    State("climate-range-clim_bio12_mean", "min"),
    State("climate-range-clim_bio12_mean", "max"),
    prevent_initial_call=True,
)
def reset_climate(n, b1_min, b1_max, b12_min, b12_max):
    if not n:
        return no_update, no_update, no_update
    # Neutral: no labels, and sliders back to full domain (not filtering)
    return [], [b1_min, b1_max], [b12_min, b12_max]

@callback(
    Output("filter-bio-level", "value", allow_duplicate=True),
    Output("filter-bio-value", "value", allow_duplicate=True),
    Output("biogeo-range-range_km2", "value", allow_duplicate=True),
    Input("btn-reset-biogeo", "n_clicks"),
    State("biogeo-range-range_km2", "min"),
    State("biogeo-range-range_km2", "max"),
    prevent_initial_call=True,
)
def reset_biogeo(n, rmin, rmax):
    if not n:
        return no_update, no_update, no_update
    return [], [], [rmin, rmax]

@callback(
    Output("bio-pct-biotype", "value", allow_duplicate=True),
    Output("bio-pct-range", "value", allow_duplicate=True),
    Input("btn-reset-biotype", "n_clicks"),
    prevent_initial_call=True,
)
def reset_biotype_pct(n):
    if not n:
        return no_update, no_update
    return None, [0, 100]

@callback(
    # taxonomy
    Output("filter-tax-kingdom", "value", allow_duplicate=True),
    Output("filter-tax-phylum", "value", allow_duplicate=True),
    Output("filter-tax-class", "value", allow_duplicate=True),
    Output("filter-tax-order", "value", allow_duplicate=True),
    Output("filter-tax-family", "value", allow_duplicate=True),
    Output("filter-tax-genus", "value", allow_duplicate=True),
    Output("filter-tax-species", "value", allow_duplicate=True),
    Output("filter-tax-id", "value", allow_duplicate=True),
    # climate + ranges
    Output("filter-climate", "value", allow_duplicate=True),
    Output("climate-range-clim_bio1_mean", "value", allow_duplicate=True),
    Output("climate-range-clim_bio12_mean", "value", allow_duplicate=True),
    # biogeo + range
    Output("filter-bio-level", "value", allow_duplicate=True),
    Output("filter-bio-value", "value", allow_duplicate=True),
    Output("biogeo-range-range_km2", "value", allow_duplicate=True),
    # biotype%
    Output("bio-pct-biotype", "value", allow_duplicate=True),
    Output("bio-pct-range", "value", allow_duplicate=True),
    # store (recomputed by D when inputs change)
    Input("btn-reset-all-filters", "n_clicks"),
    State("climate-range-clim_bio1_mean", "min"),
    State("climate-range-clim_bio1_mean", "max"),
    State("climate-range-clim_bio12_mean", "min"),
    State("climate-range-clim_bio12_mean", "max"),
    State("biogeo-range-range_km2", "min"),
    State("biogeo-range-range_km2", "max"),
    prevent_initial_call=True,
)
def reset_all(n, b1_min, b1_max, b12_min, b12_max, rk_min, rk_max):
    if not n:
        return (no_update,) * 16
    empty = []
    return (
        empty, empty, empty, empty, empty, empty, empty, empty,  # taxonomy
        empty,                                                   # climate labels
        [b1_min, b1_max], [b12_min, b12_max],                   # climate ranges
        empty, empty,                                            # biogeo level/value
        [rk_min, rk_max],                                        # distribution range
        None, [0, 100],                                          # biotype%
    )
