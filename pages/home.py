# pages/home.py
from __future__ import annotations
import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Main(
    [
        html.H1("Welcome 👋"),
        html.P(
            "This is a minimal starting point for the genome annotations dashboard. "
            "This experimental dashboard helps you explore sequenced genomes with completed annotations in our portal by location, climate, and species. "
            "You can also view gene biotype counts and examine how they vary across taxonomic groups and environments. "
            "Use the navigation above to switch pages. "
            "Use the filters above to explore the data and find species of interest. "
            "The graphs are interactive — feel free to hover and click to dive deeper."
        ),
        html.Hr(),
        html.H3("Content"),
        html.Ul(
            [
                html.Li("Home - This page"),
                html.Li("Data Browser - Find species with complete annotations from selected taxon groups, geography, and climate."),
                html.Li("Genome annotations - Explore genome annotations by taxon ranks, geography, and climate."),
            ]
        ),
        html.Hr(),
        html.H3("Give us feedback!"),
        html.P(
            "We'd love your feedback - please share it with us here! \n"
            "https://docs.google.com/forms/d/e/1FAIpQLSePe_BIU198kMf43gFIvzJFYMdJFwohfxv_sSSIl0WXh05GXQ/viewform?usp=header"
        ),
    ],
    className="page-container",
)
