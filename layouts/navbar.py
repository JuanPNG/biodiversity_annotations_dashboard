# layouts/navbar.py
from __future__ import annotations
from dash import html, dcc

def get_navbar() -> html.Header:
    links = [
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("Data Browser", href="/data-browser", className="nav-link"),
        dcc.Link("Genome Annotations", href="/genome-annotations", className="nav-link"),
    ]

    filters = html.Div(
        [
            # --- Taxonomy ranks ---
            html.Div(
                [
                    html.Label("Kingdom", className="control-label"),
                    dcc.Dropdown(id="filter-tax-kingdom", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Phylum", className="control-label"),
                    dcc.Dropdown(id="filter-tax-phylum", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Class", className="control-label"),
                    dcc.Dropdown(id="filter-tax-class", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Order", className="control-label"),
                    dcc.Dropdown(id="filter-tax-order", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Family", className="control-label"),
                    dcc.Dropdown(id="filter-tax-family", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Genus", className="control-label"),
                    dcc.Dropdown(id="filter-tax-genus", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Species", className="control-label"),
                    dcc.Dropdown(id="filter-tax-species", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            # Optionally expose tax_id as a filter too
            html.Div(
                [
                    html.Label("Tax ID", className="control-label"),
                    dcc.Dropdown(id="filter-tax-id", options=[], multi=True, placeholder="Select…"),
                ],
                className="filter-col",
            ),
            html.Div(
                html.Button("Reset taxonomy", id="btn-reset-taxonomy", n_clicks=0, className="btn-reset"),
                className="filter-col",
            ),

            html.Div([], className="filters-break"),

            html.Div(
                [
                    html.Label("Climate (categorical)", className="control-label"),
                    dcc.Dropdown(
                        id="filter-climate",
                        options=[],
                        multi=True,
                        placeholder="Select climate…",
                    ),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Biogeo level(s)", className="control-label"),
                    dcc.Dropdown(
                        id="filter-bio-level",
                        options=[],
                        multi=True,
                        placeholder="realm / biome / ecoregion…",
                    ),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Biogeo value(s)", className="control-label"),
                    dcc.Dropdown(
                        id="filter-bio-value",
                        options=[],
                        multi=True,
                        placeholder="pick regions…",
                    ),
                ],
                className="filter-col",
            ),
        ],
        className="filters-row",
    )

    return html.Header(
        [
            html.Div("Exploring Genome Annotations in Ecological Context", className="brand"),
            html.Nav(links, className="nav-links"),
            filters,
        ],
        className="navbar",
    )
