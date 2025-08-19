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
            html.Div(
                [
                    html.Label("Taxonomy", className="control-label"),
                    dcc.Dropdown(id="filter-taxonomy", options=[], multi=True, placeholder="Select taxa…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Climate (categorical)", className="control-label"),
                    dcc.Dropdown(id="filter-climate", options=[], multi=True, placeholder="Select climate…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Biogeo level(s)", className="control-label"),
                    dcc.Dropdown(id="filter-bio-level", options=[], multi=True, placeholder="realm / biome / ecoregion…"),
                ],
                className="filter-col",
            ),
            html.Div(
                [
                    html.Label("Biogeo value(s)", className="control-label"),
                    dcc.Dropdown(id="filter-bio-value", options=[], multi=True, placeholder="pick regions…"),
                ],
                className="filter-col",
            ),
        ],
        className="filters-row",
    )

    return html.Header(
        [
            html.Div("Genomes Dashboard", className="brand"),
            html.Nav(links, className="nav-links"),
            filters,
        ],
        className="navbar",
    )
