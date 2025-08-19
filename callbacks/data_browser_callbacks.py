from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback, no_update
from urllib.parse import urlparse

from utils import config
from utils.parquet_io import (
    list_columns,
    pick_default_columns,
    load_dashboard_page,
    count_dashboard_rows,  # if you added the counter; otherwise remove these 3 lines
)

def _to_markdown_link(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # ensure scheme
    href = s if s.lower().startswith(("http://", "https://")) else f"https://{s}"
    # label = hostname (fallback to href)
    try:
        host = urlparse(href).hostname or href
        label = host.replace("www.", "")
    except Exception:
        label = href
    # Markdown link
    return f"[{label}]({href})"

def _make_column_defs(df: pd.DataFrame):
    url_cols = set(getattr(config, "URL_COLUMNS", []))
    defs = []
    for col in df.columns:
        c = {"headerName": col, "field": col}
        if col in url_cols:
            c.update({
                "cellRenderer": "markdown",
                "filter": "agTextColumnFilter",
                "minWidth": 160
            })
        elif pd.api.types.is_numeric_dtype(df[col]):
            c.update({"type": "rightAligned", "filter": "agNumberColumnFilter"})
        else:
            c.update({"filter": "agTextColumnFilter"})
        defs.append(c)
    return defs


# 1) Init: preset list + columns list (do NOT set db-columns.value here)
@callback(
    Output("db-col-preset", "options"),
    Output("db-col-preset", "value"),
    Output("db-columns", "options"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_presets_and_columns(_):
    preset_names = list(config.PRESET_COLUMN_GROUPS.keys())
    preset_opts = [{"label": n.replace("_", " ").title(), "value": n} for n in preset_names]
    default_preset = (
        config.DEFAULT_COLUMN_PRESET
        if config.DEFAULT_COLUMN_PRESET in config.PRESET_COLUMN_GROUPS
        else (preset_names[0] if preset_names else None)
    )
    try:
        all_cols = list_columns(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    except Exception:
        return preset_opts, default_preset, []
    col_opts = [{"label": c, "value": c} for c in all_cols]
    return preset_opts, default_preset, col_opts

# 2) Apply preset -> sole writer to db-columns.value (runs on load and on change)
@callback(
    Output("db-columns", "value"),
    Input("db-col-preset", "value"),
    prevent_initial_call=False,  # run on first load
)
def apply_preset(preset_name):
    try:
        all_cols = list_columns(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    except Exception:
        return []
    patterns = config.PRESET_COLUMN_GROUPS.get(preset_name or "", [])
    resolved = [c for c in patterns if c in all_cols] or pick_default_columns(all_cols, max_cols=25)
    return resolved

# 3) Reset page to 1 on filter/column changes, but only if needed (avoids churn)
@callback(
    Output("db-page", "value"),
    Input("global-filters", "data"),
    Input("db-columns", "value"),
    State("db-page", "value"),
    prevent_initial_call=True,
)
def reset_page_on_change(_filters, _cols, current_page):
    return 1 if (current_page or 1) != 1 else no_update

# 4) Fetch a single page; include total count if available
@callback(
    Output("db-grid", "columnDefs"),
    Output("db-grid", "rowData"),
    Output("db-status", "children"),
    Input("global-filters", "data"),
    Input("db-page", "value"),
    Input("db-page-size", "value"),
    Input("db-columns", "value"),
    prevent_initial_call=False,
)
def update_grid(global_filters, page_value, page_size, selected_columns):
    taxonomy = (global_filters or {}).get("taxonomy") or []
    climate  = (global_filters or {}).get("climate") or []
    levels   = (global_filters or {}).get("bio_levels") or []
    values   = (global_filters or {}).get("bio_values") or []
    page = int(page_value or 1)
    size = int(page_size or 50)

    try:
        all_cols = list_columns(config.DATA_DIR / config.DASHBOARD_MAIN_FN)
    except Exception as e:
        return no_update, no_update, f"Error reading schema: {e}"

    use_cols = [c for c in (selected_columns or []) if c in all_cols] or pick_default_columns(all_cols, max_cols=25)

    url_set = set(getattr(config, "URL_COLUMNS", []) or [])
    display_cols = [c for c in use_cols if c not in url_set] + [c for c in use_cols if c in url_set]

    try:
        df, nrows = load_dashboard_page(
            columns=display_cols,
            page_number=page,
            page_size=size,
            taxonomy_filter=taxonomy,
            climate_filter=climate,
            bio_levels_filter=levels,
            bio_values_filter=values,
        )
    except Exception as e:
        return no_update, no_update, f"Error loading data: {e}"

    # Convert URL strings → Markdown links (clickable)
    url_cols = [c for c in display_cols if c in url_set and c in df.columns]
    for c in url_cols:
        df[c] = df[c].map(_to_markdown_link)

    # Reindex to the enforced order (just in case Arrow changed it)
    df = df[[c for c in display_cols if c in df.columns]]

    # Optional total count (comment out if you didn't add count_dashboard_rows)
    try:
        total = count_dashboard_rows(
            taxonomy_filter=taxonomy,
            climate_filter=climate,
            bio_levels_filter=levels,
            bio_values_filter=values,
        )
    except Exception:
        total = None

    status = f"Page {page} • size {size} • returned {nrows} rows • {len(df.columns)} columns"
    if total is not None:
        status += f" • total {total:,} rows"

    return _make_column_defs(df), df.to_dict(orient="records"), status
