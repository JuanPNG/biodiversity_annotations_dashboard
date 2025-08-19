# pages/home.py
from __future__ import annotations
import dash
from dash import html

dash.register_page(__name__, path="/", name="Home")

layout = html.Main(
    [
        html.H1("Welcome 👋"),
        html.P(
            "This is a minimal starting point for the genomes dashboard. "
            "Use the navigation above to switch pages. We’ll add data, tables, and charts step by step."
        ),
        html.Hr(),
        html.H3("What’s next"),
        html.Ul(
            [
                html.Li("Populate the global filter dropdowns from your parquet files"),
                html.Li("Add AG Grid to the Data Browser"),
                html.Li("Add charts to Genome Annotations"),
            ]
        ),
    ],
    className="page-container",
)
