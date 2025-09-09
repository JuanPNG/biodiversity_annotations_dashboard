import math

from dash import html, dcc
from utils import parquet_io, config
from utils.data_tools import gf_build_quartile_int_marks, ui_label_for_column

# Read all extents we need in a single call (cached in parquet_io)
_ENV_COLS = ["clim_bio1_mean", "clim_bio12_mean", "range_km2"]
try:
    _ENV_EXT = parquet_io.get_column_min_max(_ENV_COLS)
except Exception:
    _ENV_EXT = {c: (None, None) for c in _ENV_COLS}


def _slider_from_col(label: str | None, slider_id: str, col: str, fallback: tuple[float, float]) -> html.Div:
    """
    Build a RangeSlider initialized to the dataset's min/max for the given column.
    If `label` is None, a human-readable label is pulled from config.CLIMATE_LABELS;
    otherwise the provided label is used.
    """
    # Resolve label from central mapping when not provided
    ui_label = label or ui_label_for_column(col)

    vmin, vmax = _ENV_EXT.get(col, (None, None))
    if vmin is None or vmax is None:
        vmin, vmax = fallback

    slider_kwargs: dict = {}
    if col == "clim_bio12_mean":
        # Add clean, sparse labels
        slider_kwargs["marks"] = gf_build_quartile_int_marks(vmin, vmax)

    return html.Div(
        [
            html.Label(ui_label, className="control-label"),
            dcc.RangeSlider(
                id=slider_id,
                min=vmin,
                max=vmax,
                value=[vmin, vmax],
                allowCross=False,
                tooltip={"always_visible": False, "placement": "bottom"},
                updatemode="mouseup",
                persistence=True,
                persistence_type="session",
                **slider_kwargs,  # only adds marks for precipitation
            ),
        ],
        className="filter-col",
    )


def get_navbar() -> html.Header:
    links = [
        dcc.Link("Home", href="/", id="nav-home", className="nav-link"),
        dcc.Link("Data Browser", href="/data-browser", id="nav-data", className="nav-link"),
        dcc.Link("Genome Annotations", href="/genome-annotations", id="nav-ga", className="nav-link"),
        dcc.Link("Biotype vs Environment", href="/biotype-environment", id="nav-be", className="nav-link"),
    ]

    # --- TAXONOMY (same IDs as before) ---
    taxonomy_group = html.Details(
        [
            html.Summary(
                ["🧬 Taxonomy",
                 html.Span("0", id="tax-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Kingdom", className="control-label"),
                         dcc.Dropdown(id="filter-tax-kingdom", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Phylum", className="control-label"),
                         dcc.Dropdown(id="filter-tax-phylum", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Class", className="control-label"),
                         dcc.Dropdown(id="filter-tax-class", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Order", className="control-label"),
                         dcc.Dropdown(id="filter-tax-order", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Family", className="control-label"),
                         dcc.Dropdown(id="filter-tax-family", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Genus", className="control-label"),
                         dcc.Dropdown(id="filter-tax-genus", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Species", className="control-label"),
                         dcc.Dropdown(id="filter-tax-species", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Tax ID", className="control-label"),
                         dcc.Dropdown(id="filter-tax-id", options=[], multi=True, placeholder="Select…")],
                        className="filter-col",
                    ),
                    html.Div(
                        html.Button("Reset taxonomy", id="btn-reset-taxonomy", n_clicks=0, className="btn-reset"),
                        className="filter-col",
                    ),
                ],
                className="filters-grid",
            ),
        ],
        open=False,
        className="filter-group",
    )

    # --- CLIMATE ---
    climate_group = html.Details(
        [
            html.Summary(
                ["🌡️ Climate",
                 html.Span("0", id="climate-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Climate (categorical)", className="control-label"),
                         dcc.Dropdown(id="filter-climate", options=[], multi=True, placeholder="Select climate…")],
                        className="filter-col",
                    ),
                    _slider_from_col(None, "climate-range-clim_bio1_mean", "clim_bio1_mean", fallback=(-50.0, 50.0)),
                    _slider_from_col(None, "climate-range-clim_bio12_mean", "clim_bio12_mean", fallback=(0.0, 8000.0)),
                    html.Div(
                        html.Button("Reset climate", id="btn-reset-climate", n_clicks=0, className="btn-reset"),
                        className="filter-col",
                    ),
                ],
                className="filters-grid",
            ),
        ],
        open=False,
        className="filter-group",
    )

    # --- BIOGEOGRAPHY ---
    biogeography_group = html.Details(
        [
            html.Summary(
                ["🌍 Biogeography",
                 html.Span("0", id="biogeo-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Bioregion level(s)", className="control-label"),
                         dcc.Dropdown(id="filter-bio-level", options=[], multi=True,
                                      placeholder="realm / biome / ecoregion…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Bioregion name(s)", className="control-label"),
                         dcc.Dropdown(id="filter-bio-value", options=[], multi=True, placeholder="pick regions…")],
                        className="filter-col",
                    ),
                    _slider_from_col(None, "biogeo-range-range_km2", "range_km2", fallback=(0.0, 1_000_000.0)),
                    html.Div(
                        html.Button("Reset biogeography", id="btn-reset-biogeo", n_clicks=0, className="btn-reset"),
                        className="filter-col",
                    ),
                ],
                className="filters-grid",
            ),
        ],
        open=False,
        className="filter-group",
    )

    # --- BIOTYPE % ---
    biotype_group = html.Details(
        [
            html.Summary(
                ["🧪 Gene biotypes",
                 html.Span("0", id="bio-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Gene biotype", className="control-label"),
                         dcc.Dropdown(
                             id="bio-pct-biotype",
                             options=[], value=None, clearable=True,
                             placeholder="Choose a biotype", style={"width": "100%"},
                         )],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Proportion (%)", className="control-label"),
                         dcc.RangeSlider(
                             id="bio-pct-range",
                             min=0, max=100, value=[0, 100],
                             step=1,
                             marks={i: f"{i}%" for i in range(0, 101, 20)},
                             dots=False, allowCross=False,
                             tooltip={"always_visible": False, "placement": "bottom"},
                             updatemode="mouseup",
                         )],
                        className="filter-col filter-col--full",
                    ),
                    html.Div(
                        html.Button("Reset gene biotype", id="btn-reset-biotype", n_clicks=0, className="btn-reset"),
                        className="filter-col",
                    ),
                ],
                className="filters-grid",
            ),
        ],
        open=False,
        className="filter-group",
    )

    # --- SUPER-GROUP: Data filters (wraps groups above) ---
    reset_all_toolbar = html.Div(
        html.Button("Reset all filters", id="btn-reset-all-filters", n_clicks=0, className="btn-reset"),
        className="filters-toolbar",
    )
    data_filters_group = html.Details(
        [
            html.Summary(
                ["🔎 Data filters",
                 html.Span("0", id="all-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div([taxonomy_group, biogeography_group, climate_group, biotype_group], className="filters-grid"),
            reset_all_toolbar,
        ],
        open=True,
        className="filter-group",
    )

    # --- NAVBAR: title left; logos pinned top-right; links + filters below ---
    return html.Header(
        [
            # Head row: title + nav links (space reserved for logos via CSS)
            html.Div(
                [
                    html.Span(
                        "Exploring Genome Annotations in Ecological Context",
                        className="brand",
                    ),
                    html.Nav(links, className="nav-links"),
                ],
                className="navbar-head",
            ),

            # Filters group
            data_filters_group,

            # Logos pinned to top-right
            html.Div(
                [
                    html.Img(src="/assets/AriseLogo.png", alt="Portal logo", className="navbar-logo"),
                    html.Img(src="/assets/embl-ebi-logo.png", alt="Partner logos", className="navbar-partners"),
                ],
                className="navbar-logos",
            ),
        ],
        className="navbar",
    )

