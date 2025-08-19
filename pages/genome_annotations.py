from __future__ import annotations
import dash
from dash import html, dcc

dash.register_page(__name__, path="/genome-annotations", name="Genome Annotations")

layout = html.Main(
    [
        html.H2("Genome Annotations — Gene Biotypes"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Metric", className="control-label"),
                        dcc.Dropdown(
                            id="ga-metric",
                            options=[
                                {"label": "Percentage (mean of *_percentage)", "value": "pct"},
                                {"label": "Count (sum of *_count)", "value": "count"},
                            ],
                            value="pct",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    className="db-control db-control--narrow",
                ),
                html.Div(
                    [
                        html.Label("Biotypes", className="control-label"),
                        dcc.Dropdown(
                            id="ga-biotypes",
                            options=[],    # populated by callback (derived from schema)
                            value=[],      # empty = auto top-N
                            multi=True,
                            placeholder="Select biotypes (or leave empty for Top N)",
                            style={"width": "100%"},
                        ),
                    ],
                    className="db-control db-control--wide",
                ),
                html.Div(
                    [
                        html.Label("Top N (used when no biotypes selected)", className="control-label"),
                        dcc.Slider(
                            id="ga-topn",
                            min=5, max=50, step=1, value=15,
                            tooltip={"always_visible": False},
                        ),
                    ],
                    style={"minWidth": 280},
                    className="db-control",
                ),
            ],
            className="db-controls",
        ),
        html.Div(id="ga-status", style={"marginBottom": "8px", "opacity": 0.85}),
        dcc.Graph(id="ga-chart", figure={"data": [], "layout": {"height": 520}}),
    ],
    className="page-container",
)
