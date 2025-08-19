from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, no_update

from utils import config
from utils.parquet_io import list_biotype_columns, summarize_biotypes


# Populate biotype options from schema (strip prefix/suffix for labels)
@callback(
    Output("ga-biotypes", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_biotype_options(_):
    cols = list_biotype_columns()
    # Use pct list by default for option names; counts will share the same names
    pref = config.GENE_BIOTYPE_PREFIX
    pct_sfx = config.GENE_BIOTYPE_PCT_SUFFIX
    options = []
    for c in sorted(set(cols.get("pct", [])) | set(cols.get("count", []))):
        name = c
        if name.startswith(pref):
            name = name[len(pref):]
        # strip either suffix if present
        if name.endswith(pct_sfx):
            name = name[: -len(pct_sfx)]
        if name.endswith(config.GENE_BIOTYPE_COUNT_SUFFIX):
            name = name[: -len(config.GENE_BIOTYPE_COUNT_SUFFIX)]
        options.append({"label": name, "value": name})
    # dedupe while preserving order
    seen = set()
    deduped = [o for o in options if (o["value"] not in seen and not seen.add(o["value"]))]
    return deduped


@callback(
    Output("ga-chart", "figure"),
    Output("ga-status", "children"),
    Input("global-filters", "data"),     # taxonomy / climate / biogeo
    Input("ga-metric", "value"),         # "pct" or "count"
    Input("ga-biotypes", "value"),       # selected names (without prefix/suffix)
    Input("ga-topn", "value"),           # top N when none selected
    prevent_initial_call=False,
)
def update_biotype_chart(global_filters, metric, selected_biotypes, topn):
    metric = metric or "pct"
    sel = selected_biotypes or []
    topn = int(topn or config.BIOTYPE_TOP_N_DEFAULT)

    # Map selected names back to column names
    col_map = list_biotype_columns()
    if metric == "pct":
        all_cols = col_map.get("pct", [])
        sfx = config.GENE_BIOTYPE_PCT_SUFFIX
    else:
        all_cols = col_map.get("count", [])
        sfx = config.GENE_BIOTYPE_COUNT_SUFFIX
    pref = config.GENE_BIOTYPE_PREFIX

    def to_col(name: str) -> str:
        return f"{pref}{name}{sfx}"

    cols = [to_col(n) for n in sel] if sel else all_cols

    try:
        df = summarize_biotypes(
            metric=metric,
            biotype_cols=cols,
            taxonomy_filter=(global_filters or {}).get("taxonomy") or [],
            climate_filter=(global_filters or {}).get("climate") or [],
            bio_levels_filter=(global_filters or {}).get("bio_levels") or [],
            bio_values_filter=(global_filters or {}).get("bio_values") or [],
        )
    except Exception as e:
        return {"data": [], "layout": {"height": 520}}, f"Error summarizing biotypes: {e}"

    if df.empty:
        return {"data": [], "layout": {"height": 520}}, "No data for current filters."

    # If none selected, take top-N
    if not sel:
        df = df.head(topn)

    # Build bar chart
    title = "Gene biotypes — mean % by subset" if metric == "pct" else "Gene biotypes — total counts by subset"
    fig = px.bar(df, x="biotype", y="value", title=title)
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=60, b=60))
    fig.update_xaxes(tickangle=45)

    status = f"{len(df)} biotypes • metric={metric}"
    return fig, status
