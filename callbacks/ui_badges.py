from __future__ import annotations
from dash import Input, Output, callback

@callback(
    Output("tax-summary-badge", "children"),
    Output("env-summary-badge", "children"),
    Output("bio-summary-badge", "children"),
    Output("all-summary-badge", "children"),  # NEW: sum of the three
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
def update_filter_badges(gf):
    gf = gf or {}
    # Taxonomy: total selected values across ranks (or count ranks with selections)
    tax_map = gf.get("taxonomy_map") or {}
    tax_count = sum(len(v or []) for v in tax_map.values())

    # Environment: biogeo levels + values + climate selections
    env_count = len(gf.get("bio_levels") or []) + len(gf.get("bio_values") or []) + len(gf.get("climate") or [])

    # Biotype: 1 if a specific biotype and a non-trivial range (not 0–100)
    bp = gf.get("biotype_pct")
    if bp and bp.get("biotype"):
        rng = [float(bp.get("min", 0)), float(bp.get("max", 100))]
        bio_count = 0 if (rng[0] <= 0 and rng[1] >= 100) else 1
    else:
        bio_count = 0

    total = tax_count + env_count + bio_count
    return str(tax_count), str(env_count), str(bio_count), str(total)
