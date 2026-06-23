"""
Biotype vs Environment page layout.

This module declares controls and plots for comparing gene biotype abundance
against environmental and distribution variables:
- selected gene biotypes,
- Y-axis metric,
- optional OLS trendlines,
- optional point sizing by total genes,
- log-scale controls,
- climate X-axis plot,
- distribution X-axis plot.

Behavior is registered in callbacks/biotype_environment_callbacks.py.
Shared filters come from dcc.Store(id="global-filters"); plot controls are
local to this page.
"""

from __future__ import annotations

import dash
from dash import html, dcc
from utils import config
from utils.data_tools import ui_label_for_column

dash.register_page(
    __name__,
    path="/biotype-environment",
    name="Biotype vs Environment",
)

# X-axis variables exposed on this page.
# Add new options here only after confirming the columns exist in dashboard_main.parquet
# and have user-facing labels in utils/config.py.
CLIMATE_X_CHOICES = ["clim_bio1_mean", "clim_bio12_mean"]
DIST_X_CHOICES = ["range_km2"]  # TODO: you can add mean_elevation later


# Dash Pages supports callable layouts.
# This lets options be built when the page layout is requested.
def layout():
    # Labels
    climate_options = [{"label": ui_label_for_column(c), "value": c} for c in CLIMATE_X_CHOICES]
    dist_options = [{"label": ui_label_for_column(c), "value": c} for c in DIST_X_CHOICES]

    # --- Plot controls (collapsible) ---
    # Page-local plot controls. These change the visual encoding but do not mutate global filters.
    plot_controls_group = html.Details(
        [
            html.Summary("Plot controls", className="filter-summary"),
            html.Div(
                [
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
                                    html.Label("Log Y", className="control-label"),
                                    dcc.Checklist(
                                        id="bs-logy",
                                        options=[{"label": "", "value": "on"}],
                                        value=[],
                                        className="log-x-checklist",
                                        inputClassName="checkbox-input",
                                        labelClassName="checkbox-label",
                                        persistence=True, persistence_type="session",
                                    ),
                                ],
                                className="db-control biotype-env-log-inline",
                            ),
                        ],
                        className="biotype-env-control-row biotype-env-control-row--primary",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Climate variable (X)", className="control-label"),
                                    dcc.Dropdown(
                                        id="bs-x-climate",
                                        options=climate_options,
                                        value=CLIMATE_X_CHOICES[0],
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style={"width": "100%"},
                                    ),
                                ],
                                className="db-control db-control--wide",
                            ),
                            html.Div(
                                [
                                    html.Label("Log X", className="control-label"),
                                    dcc.Checklist(
                                        id="bs-logx-clim",
                                        options=[{"label": "", "value": "on"}],
                                        value=[],
                                        className="log-x-checklist",
                                        inputClassName="checkbox-input",
                                        labelClassName="checkbox-label",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="db-control biotype-env-log-inline",
                            ),
                            html.Div(
                                [
                                    html.Label("Distribution variable (X)", className="control-label"),
                                    dcc.Dropdown(
                                        id="bs-x-dist",
                                        options=dist_options,
                                        value=DIST_X_CHOICES[0],
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style={"width": "100%"},
                                    ),
                                ],
                                className="db-control db-control--wide",
                            ),
                            html.Div(
                                [
                                    html.Label("Log X", className="control-label"),
                                    dcc.Checklist(
                                        id="bs-logx-dist",
                                        options=[{"label": "", "value": "on"}],
                                        value=[],
                                        className="log-x-checklist",
                                        inputClassName="checkbox-input",
                                        labelClassName="checkbox-label",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="db-control biotype-env-log-inline",
                            ),
                        ],
                        className="biotype-env-control-row biotype-env-control-row--x",
                    ),
                    html.Div(
                        [
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
                        className="biotype-env-control-row biotype-env-control-row--display",
                    ),
                ],
                className="biotype-env-plot-controls",
            ),
        ],
        id="bs-plot-controls",
        open=False,                # collapsed by default
        className="filter-group",  # same styling as your other collapsibles
    )

    return html.Main(
        [
            html.H2("Biotypes vs Environment", className="home-section-title"),
            html.P(
                "Compare selected gene biotypes with climate and distribution variables. "
                "Use Plot controls to choose biotypes, change the Y-axis metric, add trendlines, and adjust point display.",
                className="prose"
            ),

            plot_controls_group,

            # Plot panels. X-variable controls live in Plot controls above.
            html.Div(
                className="biotype-env-grid",
                children=[
                    html.Section(
                        className="biotype-env-panel",
                        children=[
                            # Climate scatterplot. The callback provides the full Plotly figure.
                            dcc.Graph(
                                id="bs-fig-climate",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "520px", "width": "100%"},
                            ),
                        ],
                    ),
                    html.Section(
                        className="biotype-env-panel",
                        children=[
                            # Distribution scatterplot. It shares the same filtered row set as the climate plot.
                            dcc.Graph(
                                id="bs-fig-dist",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "520px", "width": "100%"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="bs-status", className="status-line"),
        ],
        className="page-container",
    )
