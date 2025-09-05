import math

from dash import html, dcc
from utils import parquet_io
from utils.data_tools import gf_build_quartile_int_marks

# Read all extents we need in a single call (cached in parquet_io)
_ENV_COLS = ["clim_bio1_mean", "clim_bio12_mean", "range_km2"]
try:
    _ENV_EXT = parquet_io.get_column_min_max(_ENV_COLS)
except Exception:
    _ENV_EXT = {c: (None, None) for c in _ENV_COLS}


def _slider_from_col(label: str, slider_id: str, col: str, fallback: tuple[float, float]) -> html.Div:
    """
    Build a RangeSlider initialized to the dataset's min/max for the given column.
    """
    vmin, vmax = _ENV_EXT.get(col, (None, None))
    if vmin is None or vmax is None:
        vmin, vmax = fallback

    slider_kwargs: dict = {}
    if col == "clim_bio12_mean":
        # Add clean, sparse labels
        slider_kwargs["marks"] = gf_build_quartile_int_marks(vmin, vmax)

    return html.Div(
        [
            html.Label(label, className="control-label"),
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
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("Data Browser", href="/data-browser", className="nav-link"),
        dcc.Link("Genome Annotations", href="/genome-annotations", className="nav-link"),
        dcc.Link("Biotype vs Environment", href="/biotype-environment", className="nav-link"),
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

    # --- ENVIRONMENT (biogeo + climate) ---
    # --- CLIMATE (categorical + numeric) ---
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
                    # html.Div(
                    #     [html.Label("Annual Mean Temperature (°C)", className="control-label"),
                    #      dcc.RangeSlider(
                    #          id="climate-range-clim_bio1_mean",
                    #          min=0, max=100, value=[0, 100],  # initialized by callback
                    #          allowCross=False,
                    #          tooltip={"always_visible": False, "placement": "bottom"},
                    #          updatemode="mouseup",
                    #          persistence=True, persistence_type="session",
                    #      )],
                    #     className="filter-col",
                    # ),
                    # html.Div(
                    #     [html.Label("Annual Precipitation (mm)", className="control-label"),
                    #      dcc.RangeSlider(
                    #          id="climate-range-clim_bio12_mean",
                    #          min=0, max=1000, value=[0, 1000],  # initialized by callback
                    #          allowCross=False,
                    #          tooltip={"always_visible": False, "placement": "bottom"},
                    #          updatemode="mouseup",
                    #          persistence=True, persistence_type="session",
                    #      )],
                    #     className="filter-col",
                    # ),

                    # REPLACE the two hard-coded sliders with data-driven ones:
                    _slider_from_col(
                        "Annual Mean Temperature (°C)",
                        "climate-range-clim_bio1_mean",
                        "clim_bio1_mean",
                        fallback=(-50.0, 50.0),
                    ),
                    _slider_from_col(
                        "Annual Precipitation (mm)",
                        "climate-range-clim_bio12_mean",
                        "clim_bio12_mean",
                        fallback=(0.0, 8000.0),
                    ),

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

    # --- BIOGEOGRAPHY (levels/values + distribution numeric) ---
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
                    # html.Div(
                    #     [html.Label("Distribution range size (km²)", className="control-label"),
                    #      dcc.RangeSlider(
                    #          id="biogeo-range-range_km2",
                    #          min=0, max=1_000_000, value=[0, 1_000_000],  # initialized by callback
                    #          allowCross=False,
                    #          tooltip={"always_visible": False, "placement": "bottom"},
                    #          updatemode="mouseup",
                    #          persistence=True, persistence_type="session",
                    #      )],
                    #     className="filter-col",
                    # ),

                    _slider_from_col(
                        "Distribution range size (km²)",
                        "biogeo-range-range_km2",
                        "range_km2",
                        fallback=(0.0, 1_000_000.0),
                    ),
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

    # --- BIOTYPE % (same IDs as before) ---
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
                             min=0,
                             max=100,
                             value=[0, 100],
                             step=1,
                             marks={i: f"{i}%" for i in range(0, 101, 20)},
                             dots=False,
                             allowCross=False,
                             tooltip={"always_visible": False, "placement": "bottom"},
                             updatemode="mouseup",  # update only on release (keeps callbacks light)
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

    # --- SUPER-GROUP: Data filters (wraps the three groups above) ---

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

    return html.Header(
        [
            html.Div("Exploring Genome Annotations in Ecological Context", className="brand"),
            html.Nav(links, className="nav-links"),
            # Wrap everything in a single tidy collapsible
            data_filters_group,
        ],
        className="navbar",
    )
