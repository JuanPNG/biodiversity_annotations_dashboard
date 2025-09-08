# pages/genome_annotations.py
from __future__ import annotations
import dash
from dash import html, dcc
from utils import config

dash.register_page(__name__, path="/genome-annotations", name="Genome Annotations")

_ranks = list(config.TAXONOMY_RANK_COLUMNS or [])
rank_options = [{"label": r.title(), "value": r} for r in _ranks]
DEFAULT_RANK = "kingdom" if "kingdom" in _ranks else (_ranks[0] if _ranks else None)

layout = html.Main(
    [
        html.H2("Genome Annotations — Gene Biotypes", className="home-section-title"),
        html.P(
            "This bar chart enables you to explore how the proportion of annotated gene biotypes changes between and within taxonomic groups. "
            "Use the filters above to explore the data and identify species of interest. Use the biogeography and climate filters to find species from similar environments. "
            "Select the taxon rank you wish to explore from the dropdown list below. "
            "Use the buttons to drill down and move up taxon ranks. "
            "Optionally, click on the bar of the taxon you would like to explore to drill down.",
            className="prose",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Group by rank", className="control-label"),
                        dcc.Dropdown(
                            id="ga-rank",
                            options=rank_options,     # static options from config
                            value=DEFAULT_RANK,       # default to Kingdom
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    className="db-control db-control--narrow",
                ),
                html.Div(
                    [
                        html.Button("⬆️ Up one level", id="ga-up", n_clicks=0, className="btn-reset", style={"marginLeft": "8px"}),
                        html.Button("⬇️ Down one level", id="ga-down", n_clicks=0, className="btn-reset", style={"marginLeft": "8px"}),
                        html.Button("🔄 Reset chart", id="ga-reset", n_clicks=0, className="btn-reset", style={"marginLeft": "8px"}),
                        html.Div(id="ga-crumbs", style={"opacity": 0.8, "marginTop": 6}),
                    ],
                    className="db-control",
                ),
            ],
            className="db-controls",
        ),

        html.Div("Tip: click a bar segment to drill down; use the legend to show/hide biotypes.",
                 className="prose",
                 style={"marginBottom": "6px", "opacity": 0.75}),
        html.Div(id="ga-status", style={"marginBottom": "8px", "opacity": 0.85}),
        dcc.Graph(id="ga-chart", figure={"data": [], "layout": {"height": 560}}),

        # Drill state
        dcc.Store(id="ga-drill", data={"path": []}, storage_type="memory"),
        dcc.Store(id="ga-selected-group", data={}, storage_type="memory"),
        dcc.Store(id="ga-current-groups", data=[], storage_type="memory"),
    ],
    className="page-container",
)
