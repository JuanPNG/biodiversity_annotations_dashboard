# callbacks/genome_annotations_callbacks.py
from __future__ import annotations

from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, ctx

from utils import config
from utils.parquet_io import summarize_biotypes_by_rank  # % from *_count (uses total_gene_biotypes if present)

# ---------- Drill helpers ----------
def _next_rank(current: str | None) -> str | None:
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if not current or current not in ranks:
        return None
    i = ranks.index(current)
    return ranks[i + 1] if i + 1 < len(ranks) else None

def _prev_rank(current: str | None) -> str | None:
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if not current or current not in ranks:
        return None
    i = ranks.index(current)
    return ranks[i - 1] if i - 1 >= 0 else None

# ---------- Drill handler (click bar to go deeper; button to go up) ----------
from dash import Input, Output, State, callback, no_update, ctx
from utils import config

from dash import Input, Output, State, callback, no_update, ctx
from utils import config

@callback(
    Output("ga-drill", "data", allow_duplicate=True),
    Output("ga-rank", "value", allow_duplicate=True),
    Output("ga-selected-group", "data"),
    Input("ga-chart", "clickData"),   # click → focus and drill
    Input("ga-up", "n_clicks"),       # go up one level
    Input("ga-down", "n_clicks"),     # go down one level (all children)
    Input("ga-reset", "n_clicks"),    # NEW: reset to default
    State("ga-rank", "value"),        # current rank
    State("ga-drill", "data"),
    State("ga-selected-group", "data"),
    prevent_initial_call=True,
)
def handle_drill(clickData, _up_clicks, _down_clicks, _reset_clicks, rank_value, store, selected_group):
    store = store or {"path": []}
    path = list(store.get("path", []))
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    default_rank = "kingdom" if "kingdom" in ranks else (ranks[0] if ranks else None)
    cur_rank = rank_value or default_rank
    trig = ctx.triggered_id

    def next_rank_of(r):
        if not r or r not in ranks: return None
        i = ranks.index(r)
        return ranks[i + 1] if i + 1 < len(ranks) else None

    def prev_rank_of(r):
        if not r or r not in ranks: return None
        i = ranks.index(r)
        return ranks[i - 1] if i - 1 >= 0 else r

    # CLICK: focus on clicked group and drill down one level
    if trig == "ga-chart" and clickData:
        pt = clickData["points"][0]
        group_val = pt.get("customdata") or pt.get("y") or pt.get("x")
        if group_val is None or not cur_rank:
            return no_update, no_update, no_update
        step = {"rank": cur_rank, "value": str(group_val)}
        if not path or path[-1] != step:
            path.append(step)
        nxt = next_rank_of(cur_rank)
        return {"path": path}, (nxt if nxt else no_update), step

    # UP: always move rank up one level; pop a selection if present
    if trig == "ga-up":
        prv = prev_rank_of(cur_rank)
        if not prv or prv == cur_rank:
            return no_update, no_update, no_update
        if path:
            path.pop()
        return {"path": path}, prv, no_update

    # DOWN: change grouping only (show ALL children), do not modify path
    if trig == "ga-down":
        nxt = next_rank_of(cur_rank)
        if nxt:
            return {"path": path}, nxt, no_update
        return no_update, no_update, no_update

    # RESET: clear path, jump back to default rank, clear selection
    if trig == "ga-reset":
        return {"path": []}, default_rank, {}

    return no_update, no_update, no_update


# ---------- Chart ----------
@callback(
    Output("ga-chart", "figure"),
    Output("ga-status", "children"),
    Output("ga-crumbs", "children"),
    Output("ga-current-groups", "data"),
    Input("global-filters", "data"),  # taxonomy_map + biogeo
    Input("ga-rank", "value"),
    Input("ga-drill", "data"),
    prevent_initial_call=False,
)
def update_biotype_bar(global_filters, group_rank, drill_store):
    # Fallback to kingdom if missing
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    if not group_rank:
        group_rank = "kingdom" if "kingdom" in ranks else (ranks[0] if ranks else None)

    taxonomy_map = (global_filters or {}).get("taxonomy_map") or {}
    levels       = (global_filters or {}).get("bio_levels") or []
    values       = (global_filters or {}).get("bio_values") or []

    # Apply drill path as additional taxonomy filters
    drill = (drill_store or {}).get("path", [])
    for step in drill:
        r = step.get("rank"); v = step.get("value")
        if r and v:
            taxonomy_map = {**taxonomy_map}
            taxonomy_map.setdefault(r, [])
            if v not in taxonomy_map[r]:
                taxonomy_map[r] = list(taxonomy_map[r]) + [v]

    # Summarize -> % per group from *_count (normalized by total_gene_biotypes if present)
    try:
        df = summarize_biotypes_by_rank(
            group_rank=group_rank,
            biotype_cols=None,               # include all *_count biotypes
            taxonomy_filter_map=taxonomy_map,
            climate_filter=[],               # sliders later
            bio_levels_filter=levels,
            bio_values_filter=values,
        )
    except Exception as e:
        return {"data": [], "layout": {"height": 560}}, f"Error: {e}", ""

    if df.empty:
        return {"data": [], "layout": {"height": 560}}, "No data for current selection.", "—", []

    # Build 100% stacked horizontal bars (already in %)
    pivot = df.pivot_table(index="group", columns="biotype", values="value", aggfunc="mean").fillna(0)

    groups   = pivot.index.astype(str).tolist()
    biotypes = pivot.columns.astype(str).tolist()
    height = max(360, min(900, 40 * len(groups) + 120))

    fig = go.Figure()
    for b in biotypes:
        fig.add_bar(
            x=pivot[b].values,
            y=groups,
            name=b,
            orientation="h",
            customdata=groups,  # reliable drill key
            hovertemplate="%{customdata}<br>" + f"{b}: %{{x:.2f}}%<extra></extra>",
        )

    fig.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=60, r=20, t=60, b=60),
        title=f"Gene biotype composition across {group_rank.title()}",
        legend_traceorder="normal",
        clickmode="event+select",
    )
    fig.update_xaxes(range=[0, 100], title="Percentage", ticksuffix="%")
    fig.update_yaxes(title=group_rank.title())

    crumbs = " / ".join(f"{p['rank'].title()}: {p['value']}" for p in drill) or "—"
    status = f"{len(groups)} {group_rank} • {len(biotypes)} biotypes"
    return fig, status, f"Path: {crumbs}", groups


@callback(
    Output("ga-rank", "value", allow_duplicate=True),
    Output("ga-drill", "data", allow_duplicate=True),
    Input("global-filters", "data"),
    State("ga-rank", "value"),
    prevent_initial_call=True,
)
def sync_rank_with_global_filters(global_filters, current_rank):
    """When the global taxonomy filter is changed:
       - Set GA rank to the deepest rank with selections
       - Clear the drill path so the chart shows all selected taxa at that rank
    """
    ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
    tmap = (global_filters or {}).get("taxonomy_map") or {}

    # Find deepest rank (lowest in the configured hierarchy) that has selections
    selected_ranks = [r for r in ranks if tmap.get(r)]
    if not selected_ranks:
        return no_update, no_update

    deepest = selected_ranks[-1]  # ranks are ordered top→bottom; last = deepest

    # If the rank is already correct, we still clear the drill to reflect new global selection
    new_rank = deepest if current_rank != deepest else deepest
    cleared_path = {"path": []}
    return new_rank, cleared_path
