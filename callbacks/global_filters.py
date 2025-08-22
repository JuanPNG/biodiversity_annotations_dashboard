# callbacks/global_filters.py  — consolidated + biogeo persistence fix

from __future__ import annotations
from dash import Input, Output, State, callback, no_update
from utils import config
from utils.parquet_io import list_biotype_columns
from utils.data_tools import (
    list_taxonomy_options,
    list_biogeo_levels, list_biogeo_values,
    list_taxonomy_options_cascaded,
)

# ────────────────────────────────────────────────────────────────────────────────
# Init: populate options once (do NOT blow away Biogeo values on navigation)
# ────────────────────────────────────────────────────────────────────────────────
@callback(
    Output("filter-tax-kingdom", "options"),
    Output("filter-tax-phylum", "options"),
    Output("filter-tax-class", "options"),
    Output("filter-tax-order", "options"),
    Output("filter-tax-family", "options"),
    Output("filter-tax-genus", "options"),
    Output("filter-tax-species", "options"),
    Output("filter-tax-id", "options"),
    Output("filter-bio-level", "options"),
    Output("filter-bio-value", "options"),
    Input("url", "pathname"),
    State("filter-bio-level", "value"),   # 🆕 hydrate Biogeo values options from current level(s)
    prevent_initial_call=False,
)
def init_filters(_, current_bio_levels):
    tax_opts = list_taxonomy_options()
    levels_opts = list_biogeo_levels()

    # If user already picked Biogeo level(s), keep the corresponding values’ options;
    # otherwise, leave whatever is shown today (no_update) so we don’t clear selections.
    if current_bio_levels:
        values_opts = list_biogeo_values(current_bio_levels)
    else:
        values_opts = no_update

    return (
        tax_opts.get("kingdom", []),
        tax_opts.get("phylum", []),
        tax_opts.get("class", []),
        tax_opts.get("order", []),
        tax_opts.get("family", []),
        tax_opts.get("genus", []),
        tax_opts.get("species", []),
        tax_opts.get("tax_id", []),
        levels_opts,
        values_opts,  # ← keep or hydrate, never force []
    )

# ────────────────────────────────────────────────────────────────────────────────
# Taxonomy cascade: recompute lower-rank options and prune invalid values
# ────────────────────────────────────────────────────────────────────────────────
@callback(
    # options
    Output("filter-tax-kingdom", "options", allow_duplicate=True),
    Output("filter-tax-phylum", "options", allow_duplicate=True),
    Output("filter-tax-class", "options", allow_duplicate=True),
    Output("filter-tax-order", "options", allow_duplicate=True),
    Output("filter-tax-family", "options", allow_duplicate=True),
    Output("filter-tax-genus", "options", allow_duplicate=True),
    Output("filter-tax-species", "options", allow_duplicate=True),
    Output("filter-tax-id", "options", allow_duplicate=True),
    # values (pruned)
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
    prevent_initial_call=True,
)
def cascade_taxonomy(k, p, c, o, f, g, s, taxid):
    selections = {
        "kingdom": k or [],
        "phylum": p or [],
        "class": c or [],
        "order": o or [],
        "family": f or [],
        "genus": g or [],
        "species": s or [],
        "tax_id": taxid or [],
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
        opts.get("kingdom", []),
        opts.get("phylum", []),
        opts.get("class", []),
        opts.get("order", []),
        opts.get("family", []),
        opts.get("genus", []),
        opts.get("species", []),
        opts.get("tax_id", []),
        k2, p2, c2, o2, f2, g2, s2, t2,
    )

# ────────────────────────────────────────────────────────────────────────────────
# Biogeo values options follow level(s) (unchanged), but never force-clear values
# ────────────────────────────────────────────────────────────────────────────────
@callback(
    Output("filter-bio-value", "options", allow_duplicate=True),
    Input("filter-bio-level", "value"),
    prevent_initial_call=True,
)
def update_biogeo_values(levels_value):
    # Return options for the selected levels; Dash keeps current values that still exist
    return list_biogeo_values(levels_value or [])

# ────────────────────────────────────────────────────────────────────────────────
# Persist global selections
# ────────────────────────────────────────────────────────────────────────────────
@callback(
    Output("global-filters", "data"),
    Input("filter-tax-kingdom", "value"),
    Input("filter-tax-phylum", "value"),
    Input("filter-tax-class", "value"),
    Input("filter-tax-order", "value"),
    Input("filter-tax-family", "value"),
    Input("filter-tax-genus", "value"),
    Input("filter-tax-species", "value"),
    Input("filter-tax-id", "value"),
    Input("filter-bio-level", "value"),
    Input("filter-bio-value", "value"),
    prevent_initial_call=False,
)
def store_filters(k, p, c, o, f, g, s, taxid, levels, values):
    taxonomy_map = {
        "kingdom": k or [],
        "phylum": p or [],
        "class": c or [],
        "order": o or [],
        "family": f or [],
        "genus": g or [],
        "species": s or [],
        "tax_id": taxid or [],
    }
    taxonomy_map = {rk: vs for rk, vs in taxonomy_map.items() if vs}
    return {
        "taxonomy_map": taxonomy_map,
        "bio_levels": levels or [],
        "bio_values": values or [],
        "taxonomy": [],   # legacy
        "climate": [],    # sliders later
    }

# Reset taxonomy button
@callback(
    Output("filter-tax-kingdom", "value", allow_duplicate=True),
    Output("filter-tax-phylum",  "value", allow_duplicate=True),
    Output("filter-tax-class",   "value", allow_duplicate=True),
    Output("filter-tax-order",   "value", allow_duplicate=True),
    Output("filter-tax-family",  "value", allow_duplicate=True),
    Output("filter-tax-genus",   "value", allow_duplicate=True),
    Output("filter-tax-species", "value", allow_duplicate=True),
    Output("filter-tax-id",      "value", allow_duplicate=True),
    Input("btn-reset-taxonomy", "n_clicks"),
    prevent_initial_call=True,
)
def reset_taxonomy(_n):
    empty = []
    return empty, empty, empty, empty, empty, empty, empty, empty

# ────────────────────────────────────────────────────────────────────────────────
# Gene biotype % filter: options + persist selection in global store
# ────────────────────────────────────────────────────────────────────────────────
@callback(
    Output("bio-pct-biotype", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_biotype_pct_options(_):
    col_map = list_biotype_columns()
    cnt = sorted(set(col_map.get("count", [])))
    sfx = config.GENE_BIOTYPE_COUNT_SUFFIX or "_count"
    names = [(c[:-len(sfx)] if c.endswith(sfx) else c) for c in cnt]
    seen = set()
    return [{"label": n, "value": n} for n in names if (n not in seen and not seen.add(n))]

@callback(
    Output("global-filters", "data", allow_duplicate=True),
    Input("bio-pct-biotype", "value"),
    Input("bio-pct-range", "value"),
    State("global-filters", "data"),
    prevent_initial_call=True,
)
def set_biotype_pct_in_store(biotype_name, pct_range, store):
    store = store or {}
    gf = dict(store)
    if not biotype_name or not pct_range or pct_range == [0, 100]:
        gf.pop("biotype_pct", None)
        return gf
    gf["biotype_pct"] = {
        "biotype": str(biotype_name),
        "min": float(pct_range[0]),
        "max": float(pct_range[1]),
    }
    return gf
