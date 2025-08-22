from __future__ import annotations
from dash import html, dcc

def get_navbar() -> html.Header:
    links = [
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("Data Browser", href="/data-browser", className="nav-link"),
        dcc.Link("Genome Annotations", href="/genome-annotations", className="nav-link"),
    ]

    # --- TAXONOMY (same IDs as before) ---
    taxonomy_group = html.Details(
        [
            html.Summary(
                ["🧬 Taxonomy filters ",
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
        open=True,   # open by default
        className="filter-group",
    )

    # --- ENVIRONMENT (biogeo + climate) ---
    environment_group = html.Details(
        [
            html.Summary(
                ["🌍 Environment (biogeography + climate) ",
                 html.Span("0", id="env-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Biogeo level(s)", className="control-label"),
                         dcc.Dropdown(id="filter-bio-level", options=[], multi=True,
                                      placeholder="realm / biome / ecoregion…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Biogeo value(s)", className="control-label"),
                         dcc.Dropdown(id="filter-bio-value", options=[], multi=True, placeholder="pick regions…")],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Climate (categorical)", className="control-label"),
                         dcc.Dropdown(id="filter-climate", options=[], multi=True, placeholder="Select climate…")],
                        className="filter-col",
                    ),
                    # reset just the biogeography filters (levels & values)
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
                ["🧪 Gene biotype filters ",
                 html.Span("0", id="bio-summary-badge", className="badge")],
                className="filter-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Biotype % filter", className="control-label"),
                         dcc.Dropdown(
                             id="bio-pct-biotype",
                             options=[], value=None, clearable=True,
                             placeholder="Choose a biotype", style={"width": "100%"},
                         )],
                        className="filter-col",
                    ),
                    html.Div(
                        [html.Label("Range (%)", className="control-label"),
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
                    # reset the biotype % controls
                    html.Div(
                        html.Button("Reset biotype filter", id="btn-reset-biotype", n_clicks=0, className="btn-reset"),
                        className="filter-col",
                    ),
                ],
                className="filters-grid",
            ),
        ],
        open=False,
        className="filter-group",
    )

    return html.Header(
        [
            html.Div("Exploring Genome Annotations in Ecological Context", className="brand"),
            html.Nav(links, className="nav-links"),
            html.Div([taxonomy_group, environment_group, biotype_group], className="filters-row"),
        ],
        className="navbar",
    )
