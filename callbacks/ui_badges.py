from __future__ import annotations
from dash import Input, Output, callback

@callback(
    Output("tax-summary-badge", "children"),
    Output("climate-summary-badge", "children"),
    Output("biogeo-summary-badge", "children"),
    Output("bio-summary-badge", "children"),
    Output("all-summary-badge", "children"),
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
def update_filter_badges(gf):
    gf = gf or {}

    # Taxonomy: total selected values across ranks
    tax_map = gf.get("taxonomy_map") or {}
    tax_count = sum(len(v or []) for v in tax_map.values())

    # Climate: categorical + active numeric range keys
    climate_cats = gf.get("climate") or []
    climate_ranges = gf.get("climate_ranges") or {}
    climate_count = len(climate_cats) + len(climate_ranges)

    # Biogeography: levels + values + active distribution ranges (e.g. range_km2)
    bio_levels = gf.get("bio_levels") or []
    bio_values = gf.get("bio_values") or []
    biogeo_ranges = gf.get("biogeo_ranges") or {}
    biogeo_count = len(bio_levels) + len(bio_values) + len(biogeo_ranges)

    # Biotype %: count as 1 only if a biotype is chosen AND the range is narrower than 0–100
    bp = gf.get("biotype_pct") or {}
    if bp.get("biotype"):
        lo = float(bp.get("min", 0)); hi = float(bp.get("max", 100))
        bio_pct_count = 0 if (lo <= 0 and hi >= 100) else 1
    else:
        bio_pct_count = 0

    total = tax_count + climate_count + biogeo_count + bio_pct_count
    return str(tax_count), str(climate_count), str(biogeo_count), str(bio_pct_count), str(total)