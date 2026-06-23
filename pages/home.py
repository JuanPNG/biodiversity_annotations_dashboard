"""
Home page layout.

This module declares the landing page:
- introductory text,
- filtered KPI cards,
- navigation cards for the main dashboard pages,
- feedback/contact links.

The KPI card values are populated by callbacks/home_kpis.py.
Shared filters come from dcc.Store(id="global-filters").
"""

from __future__ import annotations
import dash

dash.register_page(__name__, path="/", name="Home")

from dash import dcc, html


# Introduction
# Static introduction shown above the filtered data summaries.
intro = html.Section(
            [
                html.H1("Welcome", className="home-section-title"),
                html.Div(
                    [
                        html.P(
                            "This experimental dashboard lets you explore sequenced genomes with completed "
                            "annotations across taxonomy, climate, and geography. "
                            "You can browse gene biotype counts and see how they vary across species, taxonomic groups, "
                            "and environments. Use the filters above to refine the dataset and discover species of interest."
                        ),
                        html.P(
                            [
                                "The graphs are interactive — hover, click, and drill down to explore. ",
                                "We’d love your feedback — ",
                                html.A(
                                    "share it with us here!",
                                    href="https://docs.google.com/forms/d/1vZI2oT06ehqyheihsfVEL9Tnmz5RjxVUjLBsrqEpJX8/edit",
                                    target="_blank",
                                    rel="noopener",
                                ),
                            ]
                        ),
                    ],
                    className="home-section-body prose",
                ),
            ],
            className="home-section",
        )

# KPIs: overview of filtered data.
# KPI placeholders. Values are filled by callbacks/home_kpis.py using global filters.
kpis = html.Section(
            [
                html.H2("Key figures", className="home-section-title"),
                html.P(
                    "These key figures summarise the filtered dataset by showing how many taxa, biogeographic categories, and gene annotations are currently included after the data filters have been applied.",
                    className="home-section-body prose",
                ),
                dcc.Loading(
                    type="dot",
                    children=html.Div(
                        [
                            html.Div([html.Div("Kingdoms", className="kpi-label"), html.H3(id="kpi-kingdom")], className="kpi-card"),
                            html.Div([html.Div("Phyla", className="kpi-label"), html.H3(id="kpi-phylum")], className="kpi-card"),
                            html.Div([html.Div("Classes", className="kpi-label"), html.H3(id="kpi-class")], className="kpi-card"),
                            html.Div([html.Div("Orders", className="kpi-label"), html.H3(id="kpi-order")], className="kpi-card"),
                            html.Div([html.Div("Families", className="kpi-label"), html.H3(id="kpi-family")], className="kpi-card"),
                            html.Div([html.Div("Genera", className="kpi-label"), html.H3(id="kpi-genus")], className="kpi-card"),
                            html.Div([html.Div("Species", className="kpi-label"), html.H3(id="kpi-species")], className="kpi-card"),
                            html.Div([html.Div("Realms", className="kpi-label"), html.H3(id="kpi-bio-realm")], className="kpi-card"),
                            html.Div([html.Div("Biomes", className="kpi-label"), html.H3(id="kpi-bio-biome")], className="kpi-card"),
                            html.Div([html.Div("Ecoregions", className="kpi-label"), html.H3(id="kpi-bio-ecoregion")], className="kpi-card"),
                            html.Div([html.Div("Total annotated genes", className="kpi-label"), html.H3(id="kpi-total-genes")], className="kpi-card"),
                            html.Div(
                                [
                                    html.Div("Top gene biotypes", className="kpi-label"),
                                    html.Div(id="kpi-top-biotypes", className="kpi-subtext"),
                                ],
                                className="kpi-card kpi-card--wide",
                            ),
                        ],
                        className="kpi-grid",
                    ),
                ),
            ],
            className="home-section",
        )

# Navigation cards
# Navigation cards mirror the main navbar and help users discover dashboard sections.
nav_cards = html.Div(
    [
        dcc.Link(
            html.Div(
                [html.Div("Data Browser", className="card-title"),
                 html.Div("Take a closer look at the data by using a table with customisable fields.", className="card-sub")],
                className="nav-card"
            ),
            href="/data-browser",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Biotypes by Taxa", className="card-title"),
                 html.Div("Use stacked percentage bar charts to compare species gene biotypes within and between taxa.", className="card-sub")],
                className="nav-card"
            ),
            href="/genome-annotations",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Biotypes vs Environment", className="card-title"),
                 html.Div("Use scatter plots to examine the gene biotypes in relation to the species' environment and distribution.", className="card-sub")],
                className="nav-card"
            ),
            href="/biotype-environment",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Genome Metrics vs Environment", className="card-title"),
                 html.Div("Explore ENA genome metrics and Ensembl summaries in relation to species environment and distribution.", className="card-sub")],
                className="nav-card"
            ),
            href="/genome-metrics-environment",
            className="nav-card-link",
        ),
    ],
    className="nav-cards-grid",
)

# Pages short description and bottom navigation cards
pages_desc = html.Section(
            [
                html.H2("Explore the data", className="home-section-title"),
                html.P(
                    "Use the pages below to explore the filtered data through tables, genome annotation summaries, and environmental comparisons.",
                    className="home-section-body prose",
                ),
                nav_cards,
            ],
            className="home-section",
        )


# Page layout (three sections)
# Dash Pages reads this layout when serving the home route.
layout = html.Main(
    [
        # 1) Intro
        intro,
        # 2) KPIs
        kpis,
        # 3) Explore (bullets + nav cards)
        pages_desc,
    ],
    className="page-container",
)
