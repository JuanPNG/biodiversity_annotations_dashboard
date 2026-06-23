"""
Callbacks for the Genome Metrics vs Environment page.

This module renders two scatterplots from the same filtered dataset:
- selected genome metric vs selected climate variable,
- selected genome metric vs selected distribution variable.

It combines shared global filters with page-local plot controls. The callback
reads only the needed columns, optionally samples dense result sets, and builds
Plotly figures using the shared EMBL theme.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.dataset as ds
from dash import Input, Output, callback

from utils import config, parquet_io
from utils.data_tools import (
    get_accessions_for_biogeo,
    sizes_from_total,
    stable_sample,
    ui_label_for_column,
)
from utils.plotly_theme import apply_embl_theme

TOTAL_COL = config.TOTAL_GENES_COL


# Prepare numeric x/y arrays for trendline fitting, respecting log-axis settings.
def _prepare_xy_for_fit(
    x: np.ndarray,
    y: np.ndarray,
    logx: bool,
    logy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    x_fit = np.log10(x[mask]) if logx else x[mask]
    y_fit = np.log10(y[mask]) if logy else y[mask]
    return x_fit, y_fit


# Lightweight visual OLS trendline helper.
# This is intended for exploration, not formal statistical inference.
def _fit_line_and_curve(
    x: np.ndarray,
    y: np.ndarray,
    logx: bool,
    logy: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_fit, y_fit = _prepare_xy_for_fit(x, y, logx, logy)
    if x_fit.size < 2:
        return None

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    xs = np.linspace(x_fit.min(), x_fit.max(), 100)
    ys = intercept + slope * xs

    x_line = 10 ** xs if logx else xs
    y_line = 10 ** ys if logy else ys
    return x_line, y_line


def _read_filtered(columns: list[str], gf: dict[str, Any]) -> pd.DataFrame:
    """Read dashboard_main rows after applying global filters."""
    dset = ds.dataset(str(config.DATA_DIR / config.DASHBOARD_MAIN_FN))

    # Unpack shared filters. Missing keys are neutral/no filter.
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
    geo_rng = (gf or {}).get("biogeo_ranges") or None

    clim_expr = parquet_io._build_range_expr(dset, clim_rng)
    if clim_expr is not None:
        expr = clim_expr if expr is None else (expr & clim_expr)

    geo_expr = parquet_io._build_range_expr(dset, geo_rng)
    if geo_expr is not None:
        expr = geo_expr if expr is None else (expr & geo_expr)

    biopct = (gf or {}).get("biotype_pct") or None
    pct_expr, _ = parquet_io._build_biotype_pct_pushdown(dset, biopct)
    if pct_expr is not None:
        expr = pct_expr if expr is None else (expr & pct_expr)

    available = set(dset.schema.names)
    read_cols = [col for col in dict.fromkeys(columns) if col in available]
    if not read_cols:
        return pd.DataFrame(columns=columns)

    scanner = ds.Scanner.from_dataset(
        dset,
        columns=read_cols,
        filter=expr,
        batch_size=8192,
    )
    batches = scanner.to_batches()
    frames = [rb.to_pandas(types_mapper=pd.ArrowDtype) for rb in batches if rb.num_rows > 0]
    if not frames:
        return pd.DataFrame(columns=read_cols)

    return pd.concat(frames, ignore_index=True)


# Build one scatterplot figure for a chosen genome metric and X column.
def _make_fig(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    logx: bool,
    logy: bool,
    show_ols: bool,
    size_points: bool,
) -> go.Figure:
    fig = go.Figure()
    y_label = ui_label_for_column(ycol)

    if not xcol or not ycol or df.empty or xcol not in df.columns or ycol not in df.columns:
        fig.update_layout(
            title=title,
            height=520,
            margin=dict(l=10, r=10, t=95, b=45),
            autosize=True,
        )
        apply_embl_theme(fig)
        return fig

    keep = [xcol, ycol, "species", "accession"]
    if size_points and TOTAL_COL in df.columns:
        keep.append(TOTAL_COL)

    sub = df[keep].copy()
    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
    sub = sub.dropna(subset=[xcol, ycol])

    if sub.empty:
        fig.update_layout(
            title=title,
            height=520,
            margin=dict(l=10, r=10, t=95, b=45),
            autosize=True,
        )
        apply_embl_theme(fig)
        return fig

    show_total = size_points and TOTAL_COL in sub.columns
    total_line = "Total genes: %{customdata[2]:,.0f}<br>" if show_total else ""
    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Accession: %{customdata[1]}<br>"
        f"{total_line}"
        f"{ui_label_for_column(xcol)}: %{{x}}<br>"
        f"{y_label}: %{{y:.2f}}<extra></extra>"
    )

    if show_total:
        customdata = np.stack([sub["species"], sub["accession"], sub[TOTAL_COL]], axis=1)
    else:
        customdata = np.stack([sub["species"], sub["accession"]], axis=1)

    marker_kwargs = {}
    if show_total:
        marker_kwargs["size"] = sizes_from_total(sub[TOTAL_COL])

    fig.add_trace(
        go.Scattergl(
            x=sub[xcol],
            y=sub[ycol],
            mode="markers",
            name=y_label,
            customdata=customdata,
            hovertemplate=hover,
            marker=marker_kwargs,
        )
    )

    if show_ols:
        fit = _fit_line_and_curve(
            sub[xcol].to_numpy(),
            sub[ycol].to_numpy(),
            logx,
            logy,
        )
        if fit is not None:
            x_line, y_line = fit
            fig.add_trace(
                go.Scattergl(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="OLS",
                    showlegend=False,
                    line=dict(dash="dash", width=2),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=95, b=45),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0,
            tracegroupgap=8,
        ),
        title=dict(text=title, x=0.01, y=0.98),
        height=520,
        autosize=True,
    )
    fig.update_xaxes(
        title_text=ui_label_for_column(xcol),
        type="log" if logx else "linear",
        rangemode="tozero",
    )
    fig.update_yaxes(
        title_text=y_label,
        type="log" if logy else "linear",
        rangemode="tozero",
    )
    apply_embl_theme(fig)
    return fig


@callback(
    Output("gm-fig-climate", "figure"),
    Output("gm-fig-dist", "figure"),
    Output("gm-status", "children"),
    Input("gm-metric", "value"),
    Input("gm-x-climate", "value"),
    Input("gm-logx-clim", "value"),
    Input("gm-x-dist", "value"),
    Input("gm-logx-dist", "value"),
    Input("gm-logy", "value"),
    Input("gm-reg", "value"),
    Input("gm-size", "value"),
    Input("gm-cap", "value"),
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
# Main render callback for both scatterplots.
# Combines global filters, local plot controls, optional sampling, and Plotly
# figure construction.
def render_genome_metric_scatter(
    metric_col,
    x_clim,
    logx_clim_val,
    x_dist,
    logx_dist_val,
    logy_val,
    reg_flags,
    size_flags,
    point_cap,
    gf,
):
    gf = gf or {}

    if not metric_col:
        empty = go.Figure()
        empty.update_layout(height=520)
        apply_embl_theme(empty)
        return empty, empty, "Select a genome metric."

    size_points = "size_total" in (size_flags or [])
    show_ols = "ols" in (reg_flags or [])
    logx_clim = "on" in (logx_clim_val or [])
    logx_dist = "on" in (logx_dist_val or [])
    logy = "on" in (logy_val or [])

    # Read only columns needed for the selected metric, X variables, and point
    # sizing option.
    projected = {"species", "accession", metric_col}
    if x_clim:
        projected.add(x_clim)
    if x_dist:
        projected.add(x_dist)
    if size_points:
        projected.add(TOTAL_COL)

    df = _read_filtered(sorted(projected), gf).copy()

    cap = int(point_cap or 0)
    if cap > 0 and not df.empty:
        # Optional deterministic sampling keeps dense plots responsive without
        # changing randomly on every callback run.
        key = f"{metric_col}|{x_clim}|{x_dist}|{gf.get('climate_ranges')}|{gf.get('biogeo_ranges')}|{size_points}"
        df = stable_sample(df, cap, key)

    metric_label = ui_label_for_column(metric_col)

    fig_clim = (
        _make_fig(
            df=df,
            xcol=x_clim,
            ycol=metric_col,
            title=f"{metric_label} vs Climate",
            logx=logx_clim,
            logy=logy,
            show_ols=show_ols,
            size_points=size_points,
        )
        if x_clim
        else go.Figure()
    )

    fig_dist = (
        _make_fig(
            df=df,
            xcol=x_dist,
            ycol=metric_col,
            title=f"{metric_label} vs Distribution",
            logx=logx_dist,
            logy=logy,
            show_ols=show_ols,
            size_points=size_points,
        )
        if x_dist
        else go.Figure()
    )

    status = f"{len(df):,} rows plotted. Metric: {metric_label}"
    if logy:
        status += " • Log Y"
    if size_points:
        status += " • Point size ~ total genes"

    return fig_clim, fig_dist, status
