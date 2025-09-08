from __future__ import annotations

import dash
from dash import html, dcc
import dash_ag_grid as dag

dash.register_page(__name__, path="/data-browser", name="Data Browser")

layout = html.Main(
    [
        html.H2("Data Browser", className="home-section-title"),
        html.P("Use the filters above to explore the data and identify species of interest. "
               "Click on 'Columns & Presets' to select and change the columns that are displayed in the table. "
               "Choose the page and the number of records to be displayed. "
               "Use the filters specific to each column to narrow down your search further on the current table page.",
               className="prose"),

        # Collapsible: Columns & Presets (hidden by default)
        html.Details(
            id="db-columns-panel",
            open=False,  # hidden by default
            className="collapsible",
            children=[
                html.Summary(
                    [
                        html.Span("Columns & presets "),
                        html.Span("(0 selected)", id="db-columns-count", className="badge"),
                    ],
                    className="collapsible-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Column preset", className="control-label"),
                                dcc.Dropdown(
                                    id="db-col-preset",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    style={"width": "100%"},
                                ),
                            ],
                            className="db-control db-control--narrow",
                        ),
                        html.Div(
                            [
                                html.Label("Columns", className="control-label"),
                                dcc.Dropdown(
                                    id="db-columns",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder="Type to search columns…",
                                    style={"width": "100%"},
                                ),
                                html.Small("Tip: click and type to search."),
                            ],
                            className="db-control db-control--wide",
                        ),
                    ],
                    className="db-controls",
                ),
            ],
        ),

        html.Div(
            [
                html.Label("Page"),
                dcc.Input(id="db-page", type="number", min=1, step=1, value=1, style={"width": 100}),
                html.Label("Page size", style={"marginLeft": "12px"}),
                dcc.Dropdown(
                    id="db-page-size",
                    options=[20, 50, 100, 200, 500],
                    value=50,
                    clearable=False,
                    style={"width": 120, "display": "inline-block"},
                ),
            ],
            style={"marginBottom": "12px", "display": "flex", "alignItems": "center", "gap": "8px"},
        ),

        html.Div(id="db-status", style={"marginBottom": "8px", "opacity": 0.85}),
        dag.AgGrid(
            id="db-grid",
            columnDefs=[],
            rowData=[],
            defaultColDef={
                "sortable": True,
                "filter": True,
                "resizable": True,
                "minWidth": 110,
                "floatingFilter": True
            },
            dashGridOptions={
                "rowSelection": {"mode": "multiRow", "enableClickSelection": True},
                "rowMultiselectWithClick": True,
                "maintainColumnOrder": True,
                "alwaysShowHorizontalScroll": True,
                "wrapHeaderText": True,
                "autoHeaderHeight": True,
                "enableBrowserTooltips": True,
            },
            style={
                "height": "70vh",
                "width": "100%",
                "overflowX": "auto",
                "overflowY": "auto",
            },
            className="ag-theme-alpine-dark"
        ),
    ],
    className="page-container",
)
