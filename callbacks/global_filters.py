# callbacks/global_filters.py
from __future__ import annotations
from dash import Input, Output, callback
from utils.data_tools import get_filter_options, list_biogeo_levels, list_biogeo_values

# Init: taxonomy/climate + biogeo levels; values list starts empty
@callback(
    Output("filter-taxonomy", "options"),
    Output("filter-climate", "options"),
    Output("filter-bio-level", "options"),
    Output("filter-bio-value", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_global_filters(_):
    base = get_filter_options()        # taxonomy + climate from main parquet
    levels = list_biogeo_levels()      # levels from biogeo_long
    return base["taxonomy"], base["climate"], levels, []

# When biogeo levels change, update values list
@callback(
    Output("filter-bio-value", "options", allow_duplicate=True),  # <-- key change
    Input("filter-bio-level", "value"),
    prevent_initial_call=True,
)
def update_biogeo_values(levels_value):
    levels = levels_value or []
    return list_biogeo_values(levels)

# Store all selections
@callback(
    Output("global-filters", "data"),
    Input("filter-taxonomy", "value"),
    Input("filter-climate", "value"),
    Input("filter-bio-level", "value"),
    Input("filter-bio-value", "value"),
    prevent_initial_call=False,
)
def store_filters(tax_val, clim_val, bio_levels, bio_values):
    return {
        "taxonomy":   tax_val or [],
        "climate":    clim_val or [],
        "bio_levels": bio_levels or [],
        "bio_values": bio_values or [],
    }
