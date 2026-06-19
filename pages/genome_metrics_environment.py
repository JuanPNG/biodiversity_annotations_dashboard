from __future__ import annotations

import dash
from dash import dcc, html

from utils.data_tools import genome_metric_options, ui_label_for_column

dash.register_page(
    __name__,
    path="/genome-metrics-environment",
    name="Genome Metrics vs Environment",
)

CLIMATE_X_CHOICES = ["clim_bio1_mean", "clim_bio12_mean"]
DIST_X_CHOICES = ["range_km2"]


def layout():
    metric_options = genome_metric_options()
    default_metric = metric_options[0]["value"] if metric_options else None

    climate_options = [{"label": ui_label_for_column(c), "value": c} for c in CLIMATE_X_CHOICES]
    dist_options = [{"label": ui_label_for_column(c), "value": c} for c in DIST_X_CHOICES]

    plot_controls_group = html.Details(
        [
            html.Summary("Plot controls", className="filter-summary"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Genome metric (Y)", className="control-label"),
                                    dcc.Dropdown(
                                        id="gm-metric",
                                        options=metric_options,
                                        value=default_metric,
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
                                    html.Label("Log Y", className="control-label"),
                                    dcc.Checklist(
                                        id="gm-logy",
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
                        className="biotype-env-control-row biotype-env-control-row--primary",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Climate variable", className="control-label"),
                                    dcc.Dropdown(
                                        id="gm-x-climate",
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
                                        id="gm-logx-clim",
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
                                    html.Label("Distribution variable", className="control-label"),
                                    dcc.Dropdown(
                                        id="gm-x-dist",
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
                                        id="gm-logx-dist",
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
                                        id="gm-reg",
                                        options=[{"label": "OLS", "value": "ols"}],
                                        value=[],
                                        inputClassName="checkbox-input",
                                        labelClassName="checkbox-label",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="db-control db-control--narrow",
                            ),
                            html.Div(
                                [
                                    html.Label("Point size", className="control-label"),
                                    dcc.Checklist(
                                        id="gm-size",
                                        options=[{"label": "Size by total genes", "value": "size_total"}],
                                        value=[],
                                        inputClassName="checkbox-input",
                                        labelClassName="checkbox-label",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="db-control db-control--narrow",
                            ),
                            html.Div(
                                [
                                    html.Label("Point cap (0 = all)", className="control-label"),
                                    dcc.Input(
                                        id="gm-cap",
                                        type="number",
                                        min=0,
                                        step=1000,
                                        value=0,
                                        persistence=True,
                                        persistence_type="session",
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
        id="gm-plot-controls",
        open=True,
        className="filter-group",
    )

    return html.Main(
        [
            html.H2("Genome Metrics vs Environment", className="home-section-title"),
            html.P(
                "Explore ENA genome metrics and Ensembl summaries in relation to species climate and distribution variables.",
                className="prose",
            ),
            plot_controls_group,
            html.Div(
                className="biotype-env-grid",
                children=[
                    html.Section(
                        className="biotype-env-panel",
                        children=[
                            dcc.Graph(
                                id="gm-fig-climate",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "520px", "width": "100%"},
                            ),
                        ],
                    ),
                    html.Section(
                        className="biotype-env-panel",
                        children=[
                            dcc.Graph(
                                id="gm-fig-dist",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "520px", "width": "100%"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="gm-status", className="status-line"),
        ],
        className="page-container",
    )
