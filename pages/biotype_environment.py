# pages/biotype_environment.py
from __future__ import annotations

import dash
from dash import html, dcc
from utils import config

dash.register_page(
    __name__,
    path="/biotype-environment",
    name="Biotype vs Environment",
)

CLIMATE_X_CHOICES = ["clim_bio1_mean", "clim_bio12_mean"]
DIST_X_CHOICES = ["range_km2"]


def layout():
    return html.Main(
        [
            html.H2("Biotype vs Environment"),
            html.P(
                "Explore relationships between gene biotype and climate/distribution. "
                "All global filters (taxonomy, biogeography, climate, numeric ranges) apply."
            ),

            # Controls row
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Biotypes (Y)", className="control-label"),
                            dcc.Dropdown(
                                id="bs-biotypes",
                                options=[],   # populated by callback
                                value=[],     # populated by callback
                                multi=True,
                                placeholder="Choose 1–6 biotypes…",
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                        ],
                        className="db-control db-control--wide",
                    ),

                    html.Div(
                        [
                            html.Label("Y metric", className="control-label"),
                            dcc.RadioItems(
                                id="bs-y-metric",
                                options=[
                                    {"label": "Per-1k genes", "value": "per1k"},
                                    {"label": "Percentage",   "value": "percentage"},
                                    {"label": "Raw count",    "value": "raw"},
                                ],
                                value="per1k",
                                inputClassName="radio-input",
                                labelClassName="radio-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),

                    html.Div(
                        [
                            html.Label("Climate X", className="control-label"),
                            dcc.Dropdown(
                                id="bs-x-climate",
                                options=[{"label": c, "value": c} for c in CLIMATE_X_CHOICES],
                                value=CLIMATE_X_CHOICES[0],
                                clearable=False,
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),
                    html.Div(
                        [
                            html.Label("Distribution X", className="control-label"),
                            dcc.Dropdown(
                                id="bs-x-dist",
                                options=[{"label": c, "value": c} for c in DIST_X_CHOICES],
                                value=DIST_X_CHOICES[0],
                                clearable=False,
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),

                    html.Div(
                        [
                            html.Label("Log axes", className="control-label"),
                            dcc.Checklist(
                                id="bs-log",
                                options=[
                                    {"label": "Log Climate X", "value": "log_x_clim"},
                                    {"label": "Log Dist X", "value": "log_x_dist"},
                                    {"label": "Log Y", "value": "log_y"},
                                ],
                                value=[],
                                inputClassName="checkbox-input",
                                labelClassName="checkbox-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),

                    html.Div(
                        [
                            html.Label("Trendlines", className="control-label"),
                            dcc.Checklist(
                                id="bs-reg",
                                options=[{"label": "OLS per biotype", "value": "ols"}],
                                value=[],
                                inputClassName="checkbox-input",
                                labelClassName="checkbox-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),

                    html.Div(
                        [
                            html.Label("Point size", className="control-label"),
                            dcc.Checklist(
                                id="bs-size",
                                options=[{"label": "Size by total genes", "value": "size_total"}],
                                value=[],
                                inputClassName="checkbox-input",
                                labelClassName="checkbox-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),

                    html.Div(
                        [
                            html.Label("Point cap (0 = all)", className="control-label"),
                            dcc.Input(
                                id="bs-cap",
                                type="number",
                                min=0,
                                step=1000,
                                value=0,
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                        ],
                        className="db-control db-control--narrow",
                    ),
                ],
                className="db-controls",
            ),

            html.Div(className="charts-grid two", children=[
                dcc.Graph(id="bs-fig-climate", config={"displayModeBar": False}),
                dcc.Graph(id="bs-fig-dist", config={"displayModeBar": False}),
            ]),
            html.Div(id="bs-status", className="status-line"),
        ],
        className="page-container",
    )
