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

# X options (feel free to extend later)
CLIMATE_X_CHOICES = ["clim_bio1_mean", "clim_bio12_mean"]
DIST_X_CHOICES = ["range_km2"]  # you can add mean_elevation later


def layout():
    # Panel A: Climate (controls on the side)
    climate_options = [{"label": config.CLIMATE_LABELS.get(c, c), "value": c} for c in CLIMATE_X_CHOICES]
    # --- Plot controls (collapsible) ---
    plot_controls_group = html.Details(
        [
            html.Summary("Plot controls", className="filter-summary"),
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
                            html.Label("Log Y", className="control-label"),
                            dcc.Checklist(
                                id="bs-logy",
                                options=[{"label": "Enable", "value": "on"}],
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
                className="db-controls",  # reuse your existing row styling inside the collapsible
            ),
        ],
        open=False,                # collapsed by default
        className="filter-group",  # same styling as your other collapsibles
    )

    return html.Main(
        [
            html.H2("Biotype vs Environment"),
            html.P(
                "Explore relationships between gene biotype abundance and climate/distribution. "
                "All global filters (taxonomy, biogeography, climate, numeric ranges, biotype%) apply."
            ),

            plot_controls_group,

            # Panel A: Climate (controls on the side)
            html.Div(
                className="viz-split",
                children=[
                    html.Div(
                        className="panel",
                        children=[
                            html.H4("Climate X", className="panel-title"),
                            dcc.Dropdown(
                                id="bs-x-climate",
                                options=climate_options,   # <— use friendly labels
                                value=CLIMATE_X_CHOICES[0],
                                clearable=False,
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                            html.Div(style={"height": 8}),
                            html.Label("Log X", className="control-label"),
                            dcc.Checklist(
                                id="bs-logx-clim",
                                options=[{"label": "Enable", "value": "on"}],
                                value=[],  # unchecked by default
                                inputClassName="checkbox-input",
                                labelClassName="checkbox-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="bs-fig-climate",
                        config={"displayModeBar": False, "responsive": False},
                        style={"height": "520px", "width": "800px"},
                    ),
                ],
            ),

            # Panel B: Distribution (controls on the side)
            html.Div(
                className="viz-split",
                children=[
                    html.Div(
                        className="panel",
                        children=[
                            html.H4("Distribution X", className="panel-title"),
                            dcc.Dropdown(
                                id="bs-x-dist",
                                options=[{"label": c, "value": c} for c in DIST_X_CHOICES],
                                value=DIST_X_CHOICES[0],
                                clearable=False,
                                persistence=True, persistence_type="session",
                                style={"width": "100%"},
                            ),
                            html.Div(style={"height": 8}),
                            html.Label("Log X", className="control-label"),
                            dcc.Checklist(
                                id="bs-logx-dist",
                                options=[{"label": "Enable", "value": "on"}],
                                value=[],  # unchecked by default
                                inputClassName="checkbox-input",
                                labelClassName="checkbox-label",
                                persistence=True, persistence_type="session",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="bs-fig-dist",
                        config={"displayModeBar": False, "responsive": False},
                        style={"height": "520px", "width": "800px"},
                    ),
                ],
            ),

            html.Div(id="bs-status", className="status-line"),
        ],
        className="page-container",
    )
