# pages/home.py
from __future__ import annotations
import dash

dash.register_page(__name__, path="/", name="Home")

from dash import dcc, html

nav_cards = html.Div(
    [
        dcc.Link(
            html.Div(
                [html.Div("Data Browser", className="card-title"),
                 html.Div("Browse and filter rows with server-side paging.", className="card-sub")],
                className="nav-card"
            ),
            href="/data-browser",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Genome Annotations", className="card-title"),
                 html.Div("Stacked % by biotype with drilldown.", className="card-sub")],
                className="nav-card"
            ),
            href="/genome-annotations",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Biotype vs Environment", className="card-title"),
                 html.Div("Scatterplots of biotypes against climate and distribution.", className="card-sub")],
                className="nav-card"
            ),
            href="/biotype-environment",
            className="nav-card-link",
        ),
        dcc.Link(
            html.Div(
                [html.Div("Maps (soon)", className="card-title"),
                 html.Div("GBIF occurrences with clustering.", className="card-sub")],
                className="nav-card nav-card--disabled"
            ),
            href="#",
            className="nav-card-link",
        ),
    ],
    className="nav-cards-grid",
)

# then include `nav_cards` in the layout where you want it:
# layout = html.Main([... existing content ..., nav_cards, ...])



layout = html.Main(
    [
        html.H1("Welcome 👋"),
        html.P(
            "Explore sequenced genomes and their annotations across taxonomy and environment. "
            "Use the filters above to refine the dataset; metrics below reflect the current selections."
        ),

        # KPIs — numbers only, filled by callbacks
        dcc.Loading(
            type="dot",
            children=html.Div(
                [
                    html.Div([html.Div("Kingdoms", className="kpi-label"), html.H3(id="kpi-kingdom")],
                             className="kpi-card"),
                    html.Div([html.Div("Phyla", className="kpi-label"), html.H3(id="kpi-phylum")],
                             className="kpi-card"),
                    html.Div([html.Div("Classes", className="kpi-label"), html.H3(id="kpi-class")],
                             className="kpi-card"),
                    html.Div([html.Div("Orders", className="kpi-label"), html.H3(id="kpi-order")],
                             className="kpi-card"),
                    html.Div([html.Div("Families", className="kpi-label"), html.H3(id="kpi-family")],
                             className="kpi-card"),
                    html.Div([html.Div("Genera", className="kpi-label"), html.H3(id="kpi-genus")],
                             className="kpi-card"),
                    html.Div([html.Div("Species", className="kpi-label"), html.H3(id="kpi-species")],
                             className="kpi-card"),

                    html.Div([html.Div("Realms", className="kpi-label"), html.H3(id="kpi-bio-realm")],
                             className="kpi-card"),
                    html.Div([html.Div("Biomes", className="kpi-label"), html.H3(id="kpi-bio-biome")],
                             className="kpi-card"),
                    html.Div([html.Div("Ecoregions", className="kpi-label"), html.H3(id="kpi-bio-ecoregion")],
                             className="kpi-card"),

                    html.Div([html.Div("Total annotated genes", className="kpi-label"), html.H3(id="kpi-total-genes")],
                             className="kpi-card"),
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

        html.Hr(),
        html.H3("Content"),
        html.Ul(
            [
                html.Li("Home — Overview & KPIs reflecting active filters."),
                html.Li("Data Browser — Table with server-side paging and column presets."),
                html.Li("Genome Annotations — Stacked % bars with drill up/down."),
                html.Li("Biotype vs Environment — Scatterplots linking biotypes with climate and distribution."),
            ]
        ),
        nav_cards
    ],
    className="page-container",
)