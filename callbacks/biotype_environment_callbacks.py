from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback
import pyarrow.dataset as ds

from utils import config
from utils import parquet_io
from utils.data_tools import (
    biotype_pct_columns,
    pct_to_count,
    sizes_from_total,
    stable_sample,
    get_accessions_for_biogeo,
)

PCT_SUFFIX   = config.GENE_BIOTYPE_PCT_SUFFIX
COUNT_SUFFIX = config.GENE_BIOTYPE_COUNT_SUFFIX
TOTAL_COL    = config.TOTAL_GENES_COL
EXCLUDE_SET  = set(config.GENE_BIOTYPE_EXCLUDE or ())


def _prepare_xy_for_fit(x: np.ndarray, y: np.ndarray, logx: bool, logy: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    X = np.log10(x[mask]) if logx else x[mask]
    Y = np.log10(y[mask]) if logy else y[mask]
    return X, Y

def _fit_line_and_curve(x: np.ndarray, y: np.ndarray, logx: bool, logy: bool) -> tuple[np.ndarray, np.ndarray] | None:
    X, Y = _prepare_xy_for_fit(x, y, logx, logy)
    if X.size < 2:
        return None
    b, a = np.polyfit(X, Y, 1)  # slope, intercept
    xs = np.linspace(X.min(), X.max(), 100)
    ys = a + b * xs
    x_line = (10 ** xs) if logx else xs
    y_line = (10 ** ys) if logy else ys
    return x_line, y_line

def _read_filtered(columns: List[str], gf: dict) -> pd.DataFrame:
    dset = ds.dataset(str(config.DATA_DIR / config.DASHBOARD_MAIN_FN))

    tax_map = (gf or {}).get("taxonomy_map") or {}
    climate_cats = (gf or {}).get("climate") or []
    levels = (gf or {}).get("bio_levels") or []
    values = (gf or {}).get("bio_values") or []

    accession_filter = get_accessions_for_biogeo(levels, values)

    expr = parquet_io._build_filter_expr(
        dset=dset,
        taxonomy_filter=None,
        taxonomy_filter_map=tax_map,
        climate_filter=climate_cats,
        accession_filter=accession_filter or None,
    )

    clim_rng = (gf or {}).get("climate_ranges") or None
    geo_rng  = (gf or {}).get("biogeo_ranges") or None
    r1 = parquet_io._build_range_expr(dset, clim_rng)
    if r1 is not None:
        expr = r1 if expr is None else (expr & r1)
    r2 = parquet_io._build_range_expr(dset, geo_rng)
    if r2 is not None:
        expr = r2 if expr is None else (expr & r2)

    # Global Gene Biotype% pushdown
    biopct = (gf or {}).get("biotype_pct") or None
    pct_expr, _ = parquet_io._build_biotype_pct_pushdown(dset, biopct)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    scanner = ds.Scanner.from_dataset(dset, columns=columns, filter=expr, batch_size=8192)
    batches = scanner.to_batches()
    frames = [rb.to_pandas(types_mapper=pd.ArrowDtype) for rb in batches if rb.num_rows > 0]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)

def _make_fig(
    df: pd.DataFrame,
    xcol: str,
    biotype_cols: list[str],
    title: str,
    logx: bool,
    logy: bool,
    show_ols: bool,
    y_axis_title: str,
    hover_suffix: str,
    size_points: bool,
) -> go.Figure:
    fig = go.Figure()
    if not xcol or df.empty:
        fig.update_layout(template="plotly_dark", title=title, height=520, width=800, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    for c in biotype_cols:
        if c not in df.columns:
            continue
        keep = [xcol, c, "species", "accession"]
        if size_points and TOTAL_COL in df.columns:
            keep.append(TOTAL_COL)
        sub = df[keep].copy()
        sub = sub.dropna(subset=[xcol, c])
        if sub.empty:
            continue

        show_total = size_points and (TOTAL_COL in sub.columns)
        total_line = "Total genes: %{customdata[2]:,.0f}<br>" if show_total else ""
        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Accession: %{customdata[1]}<br>"
            f"{total_line}"
            f"{xcol}: %{{x}}<br>%{{y:.2f}}{hover_suffix}<extra></extra>"
        )

        if show_total:
            cd = np.stack([sub["species"], sub["accession"], sub[TOTAL_COL]], axis=1)
        else:
            cd = np.stack([sub["species"], sub["accession"]], axis=1)

        marker_kwargs = {}
        if show_total:
            marker_kwargs["size"] = sizes_from_total(sub[TOTAL_COL])

        yvals = pd.to_numeric(sub[c], errors="coerce")
        fig.add_trace(go.Scattergl(
            x=sub[xcol],
            y=yvals,
            mode="markers",
            name=c.removesuffix(PCT_SUFFIX),
            legendgroup=c,
            customdata=cd,
            hovertemplate=hover,
            marker=marker_kwargs,
        ))

        if show_ols:
            fit = _fit_line_and_curve(sub[xcol].to_numpy(), yvals.to_numpy(), logx, logy)
            if fit is not None:
                x_line, y_line = fit
                fig.add_trace(go.Scattergl(
                    x=x_line, y=y_line,
                    mode="lines",
                    name=c.removesuffix(PCT_SUFFIX) + " (OLS)",
                    legendgroup=c,
                    showlegend=False,
                    line=dict(dash="dash", width=2),
                    hoverinfo="skip",
                ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=70),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            tracegroupgap=8,
        ),
        title=dict(text=title, x=0.01, y=0.98),
        height=520,
        autosize=False,
    )
    fig.update_xaxes(title_text=xcol, type=("log" if logx else "linear"), rangemode="tozero")
    fig.update_yaxes(title_text=y_axis_title, type=("log" if logy else "linear"), rangemode="tozero")
    return fig


@callback(
    Output("bs-biotypes", "options"),
    Output("bs-biotypes", "value"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def init_biotypes(_):
    cols = biotype_pct_columns()
    opts = [{"label": c.removesuffix(PCT_SUFFIX), "value": c} for c in cols]
    return opts, [o["value"] for o in opts[:4]]  # default up to 4 biotypes


@callback(
    Output("bs-fig-climate", "figure"),
    Output("bs-fig-dist", "figure"),
    Output("bs-status", "children"),
    Input("bs-biotypes", "value"),
    Input("bs-y-metric", "value"),
    Input("bs-reg", "value"),
    Input("bs-size", "value"),
    Input("bs-cap", "value"),
    Input("bs-x-climate", "value"),
    Input("bs-logx-clim", "value"),
    Input("bs-x-dist", "value"),
    Input("bs-logx-dist", "value"),
    Input("bs-logy", "value"),
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
def render_scatter(biotype_cols, y_metric, reg_flags, size_flags, point_cap,
                   x_clim, logx_clim_val, x_dist, logx_dist_val, logy_val, gf):

    biotype_cols = list(biotype_cols or [])
    if not biotype_cols:
        empty = go.Figure(); empty.update_layout(template="plotly_dark", height=520, width=800)
        return empty, empty, "Select at least one biotype."

    # Which Y values to read & label
    if y_metric == "percentage":
        y_cols_to_project = biotype_cols[:]  # *_percentage
        y_axis_title = "Gene biotype (%)"
        hover_suffix = "%"
    elif y_metric == "raw":
        y_cols_to_project = [pct_to_count(c) for c in biotype_cols]  # *_count
        y_axis_title = "Gene biotype (count)"
        hover_suffix = ""
    else:  # per1k
        y_cols_to_project = [pct_to_count(c) for c in biotype_cols] + [TOTAL_COL]
        y_axis_title = "Gene biotype (per 1k genes)"
        hover_suffix = " per 1k genes"

    # Projection
    proj = {"species", "accession"}
    if x_clim: proj.add(x_clim)
    if x_dist: proj.add(x_dist)
    proj.update(y_cols_to_project)

    # Always include TOTAL_COL if point sizing is requested
    size_points = "size_total" in (size_flags or [])
    if size_points:
        proj.add(TOTAL_COL)

    df = _read_filtered(sorted(proj), gf).copy()

    # Materialize chosen Y metric into *_percentage-named columns for uniform plotting
    def _materialize_y(df_in: pd.DataFrame, cols_pct: list[str], metric: str) -> tuple[pd.DataFrame, list[str]]:
        df_out = df_in.copy()
        keep: list[str] = []
        if metric == "percentage":
            for c in cols_pct:
                if c in df_out.columns:
                    keep.append(c)
            return df_out, keep

        for c_pct in cols_pct:
            c_cnt = pct_to_count(c_pct)
            if c_cnt not in df_out.columns:
                continue
            if metric == "raw":
                df_out[c_pct] = pd.to_numeric(df_out[c_cnt], errors="coerce")
            else:  # per1k
                denom = pd.to_numeric(df_out.get(TOTAL_COL), errors="coerce")
                num   = pd.to_numeric(df_out[c_cnt], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    per1k = 1000.0 * (num / denom)
                df_out[c_pct] = per1k.replace([np.inf, -np.inf], np.nan)
            keep.append(c_pct)
        return df_out, keep

    df, biotype_cols = _materialize_y(df, biotype_cols, y_metric)

    # Stable optional sampling
    cap = int(point_cap or 0)
    if cap > 0 and not df.empty:
        key = f"{sorted(biotype_cols)}|{x_clim}|{x_dist}|{gf.get('climate_ranges')}|{gf.get('biogeo_ranges')}|{y_metric}|{size_points}"
        df = stable_sample(df, cap, key)

    # Flags
    logx_clim = "on" in (logx_clim_val or [])
    logx_dist = "on" in (logx_dist_val or [])
    logy      = "on" in (logy_val or [])
    show_ols  = "ols" in (reg_flags or [])

    fig_clim = _make_fig(
        df, x_clim, biotype_cols, "Biotypes vs Climate",
        logx_clim, logy, show_ols, y_axis_title, hover_suffix, size_points
    ) if x_clim else go.Figure()

    fig_dist = _make_fig(
        df, x_dist, biotype_cols, "Biotypes vs Distribution",
        logx_dist, logy, show_ols, y_axis_title, hover_suffix, size_points
    ) if x_dist else go.Figure()

    nrows = len(df)
    status = f"{nrows:,} rows plotted; traces = {len(biotype_cols)} biotypes. Metric: {y_metric}"
    if y_metric == "raw" and not logy:
        status += " — Tip: raw counts often benefit from Log Y."
    if size_points:
        status += " — Point size ~ total genes."
    return fig_clim, fig_dist, status
