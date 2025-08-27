from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback
import pyarrow.dataset as ds

from utils import config
from utils import parquet_io

PCT_SUFFIX   = config.GENE_BIOTYPE_PCT_SUFFIX         # e.g. "_percentage"
COUNT_SUFFIX = config.GENE_BIOTYPE_COUNT_SUFFIX       # e.g. "_count"
TOTAL_COL    = config.TOTAL_GENES_COL                 # "total_gene_biotypes"
EXCLUDE_SET  = set(config.GENE_BIOTYPE_EXCLUDE or ())


def _discover_biotype_pct_columns() -> list[str]:
    """Find *_percentage columns (excluding configured items)."""
    dataset = ds.dataset(str(config.DATA_DIR / config.DASHBOARD_MAIN_FN))
    names = list(dataset.schema.names)
    return [c for c in names if c.endswith(PCT_SUFFIX) and c not in EXCLUDE_SET]


def _get_accessions_for_biogeo(levels: list[str], values: list[str]) -> list[str]:
    """
    Resolve selected biogeo levels/values to accession allow-list from biogeo_long.
    If no levels/values selected, returns [] (meaning 'no accession filter').
    """
    if not levels and not values:
        return []
    dset = ds.dataset(str(config.DATA_DIR / config.BIOGEO_LONG_FN))
    expr = None
    if levels:
        e = ds.field(config.BIOGEO_LEVEL_COL).isin(levels)
        expr = e if expr is None else (expr & e)
    if values:
        e = ds.field(config.BIOGEO_VALUE_COL).isin(values)
        expr = e if expr is None else (expr & e)
    scanner = ds.Scanner.from_dataset(
        dset,
        columns=[config.ACCESSION_COL_BIOGEO],
        filter=expr,
        batch_size=8192,
    )
    batches = scanner.to_batches()
    if not batches:
        return []
    sers = [
        rb.to_pandas()[config.ACCESSION_COL_BIOGEO]
        for rb in batches if rb.num_rows > 0
    ]
    if not sers:
        return []
    acc = pd.concat(sers, ignore_index=True)
    if acc.empty:
        return []
    return sorted(pd.unique(acc.dropna().astype(str)).tolist())


def _stable_sample(df: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    """Seeded reservoir-like sample for stable visuals."""
    if n <= 0 or len(df) <= n:
        return df
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


def _prepare_xy_for_fit(x: np.ndarray, y: np.ndarray, logx: bool, logy: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed X,Y for linear fit; drop NaN/Inf and non-positive where logs are requested."""
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
    """
    Fit Y ~ a + b*X (in log space if toggled), and return (x_line, y_line) in ORIGINAL space.
    Returns None if not enough points.
    """
    X, Y = _prepare_xy_for_fit(x, y, logx, logy)
    if X.size < 2:
        return None
    b, a = np.polyfit(X, Y, 1)  # slope, intercept
    xs = np.linspace(X.min(), X.max(), 100)
    ys = a + b * xs
    x_line = (10 ** xs) if logx else xs
    y_line = (10 ** ys) if logy else ys
    return x_line, y_line


def _sizes_from_total(total: pd.Series) -> np.ndarray:
    """Map total genes to marker sizes (px) using robust sqrt scaling."""
    t = pd.to_numeric(total, errors="coerce").to_numpy(dtype=float)
    t[np.isinf(t)] = np.nan
    if np.all(~np.isfinite(t)):
        return np.full_like(t, 8.0, dtype=float)
    finite = t[np.isfinite(t)]
    if finite.size < 2:
        return np.full_like(t, 8.0, dtype=float)
    # robust 5th–95th pct range
    q_lo, q_hi = np.nanquantile(finite, [0.05, 0.95])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        q_lo, q_hi = np.nanmin(finite), np.nanmax(finite)
        if not np.isfinite(q_hi) or q_hi <= q_lo:
            return np.full_like(t, 8.0, dtype=float)
    norm = np.clip((t - q_lo) / max(q_hi - q_lo, 1.0), 0.0, 1.0)
    return 6.0 + 10.0 * np.sqrt(norm)  # 6–16 px


def _pct_to_count(col_pct: str) -> str:
    base = col_pct.removesuffix(PCT_SUFFIX)
    return f"{config.GENE_BIOTYPE_PREFIX}{base}{COUNT_SUFFIX}"


def _read_filtered(columns: List[str], gf: dict) -> pd.DataFrame:
    """
    Read a projected, filtered slice from dashboard_main using:
    - taxonomy_map + climate categorical via _build_filter_expr
    - accession allow-list for biogeo levels/values
    - numeric ranges (climate_ranges + biogeo_ranges)
    """
    dset = ds.dataset(str(config.DATA_DIR / config.DASHBOARD_MAIN_FN))

    tax_map = (gf or {}).get("taxonomy_map") or {}
    climate_cats = (gf or {}).get("climate") or []
    levels = (gf or {}).get("bio_levels") or []
    values = (gf or {}).get("bio_values") or []

    accession_filter = _get_accessions_for_biogeo(levels, values)

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

    # Global Gene Biotype% range pushdown (e.g., {"biotype": "protein_coding", "min": 10, "max": 40})
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
        fig.update_layout(template="plotly_dark", title=title, height=480, margin=dict(l=10, r=10, t=40, b=10))
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

        # Hover template
        show_total = size_points and (TOTAL_COL in sub.columns)
        total_line = "Total genes: %{customdata[2]:,.0f}<br>" if show_total else ""
        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Accession: %{customdata[1]}<br>"
            f"{total_line}"
            f"{xcol}: %{{x}}<br>%{{y:.2f}}{hover_suffix}<extra></extra>"
        )

        # Customdata (species, accession, [total_genes?])
        if show_total:
            cd = np.stack([sub["species"], sub["accession"], sub[TOTAL_COL]], axis=1)
        else:
            cd = np.stack([sub["species"], sub["accession"]], axis=1)

        # Marker sizes
        marker_kwargs = {}
        if show_total:
            marker_kwargs["size"] = _sizes_from_total(sub[TOTAL_COL])

        # Points
        yvals = pd.to_numeric(sub[c], errors="coerce")
        fig.add_trace(go.Scattergl(
            x=sub[xcol],
            y=yvals,
            mode="markers",
            name=c.removesuffix(PCT_SUFFIX),
            legendgroup=c,
            customdata=cd,
            hovertemplate=hover,
            marker=marker_kwargs,  # applies size array if present
        ))

        # Optional OLS line on displayed data
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
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        title=title,
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
    cols = _discover_biotype_pct_columns()
    opts = [{"label": c.removesuffix(PCT_SUFFIX), "value": c} for c in cols]
    # default up to 4 to reduce clutter
    return opts, [o["value"] for o in opts[:4]]


@callback(
    Output("bs-fig-climate", "figure"),
    Output("bs-fig-dist", "figure"),
    Output("bs-status", "children"),
    Input("bs-biotypes", "value"),
    Input("bs-x-climate", "value"),
    Input("bs-x-dist", "value"),
    Input("bs-log", "value"),
    Input("bs-cap", "value"),
    Input("bs-reg", "value"),
    Input("bs-y-metric", "value"),
    Input("bs-size", "value"),
    Input("global-filters", "data"),
    prevent_initial_call=False,
)
def render_scatter(biotype_cols, x_clim, x_dist, log_flags, point_cap, reg_flags, y_metric, size_flags, gf):
    biotype_cols = list(biotype_cols or [])
    if not biotype_cols:
        empty = go.Figure(); empty.update_layout(template="plotly_dark", height=480)
        return empty, empty, "Select at least one biotype."

    # Decide which columns to project and how to label Y/hover
    if y_metric == "percentage":
        y_cols_to_project = biotype_cols[:]                      # *_percentage
        y_axis_title = "Gene biotype (%)"
        hover_suffix = "%"
    elif y_metric == "raw":
        y_cols_to_project = [_pct_to_count(c) for c in biotype_cols]  # *_count
        y_axis_title = "Gene biotype (count)"
        hover_suffix = ""
    else:  # per1k
        y_cols_to_project = [_pct_to_count(c) for c in biotype_cols] + [TOTAL_COL]
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

    # Materialize chosen Y metric into the *_percentage-named columns so plotting code stays uniform
    def _materialize_y(df_in: pd.DataFrame, cols_pct: list[str], metric: str) -> tuple[pd.DataFrame, list[str]]:
        df_out = df_in.copy()
        keep: list[str] = []
        if metric == "percentage":
            for c in cols_pct:
                if c in df_out.columns:
                    keep.append(c)
            return df_out, keep

        for c_pct in cols_pct:
            c_cnt = _pct_to_count(c_pct)
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

    # Optional stable sampling
    cap = int(point_cap or 0)
    if cap > 0 and not df.empty:
        key = f"{sorted(biotype_cols)}|{x_clim}|{x_dist}|{gf.get('climate_ranges')}|{gf.get('biogeo_ranges')}|{y_metric}|{size_points}"
        df = _stable_sample(df, cap, key)

    logx_clim = "log_x_clim" in (log_flags or [])
    logx_dist = "log_x_dist" in (log_flags or [])
    logy      = "log_y" in (log_flags or [])
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
