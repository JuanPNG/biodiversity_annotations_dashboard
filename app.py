from __future__ import annotations

from dash import Dash, html, dcc
import dash
from layouts.navbar import get_navbar

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Exploring Genome Annotations in Ecological Context",
)

server = app.server

# --- Main shell ---
app_shell = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="global-filters", storage_type="memory"),
        get_navbar(),
        dash.page_container,
    ],
    className="app-root",
)

app.layout = app_shell

# --- Validation layout: include page layouts so callback IDs are known ---
from pages import home as _home
from pages import data_browser as _data_browser
from pages import genome_annotations as _genome_annotations

app.validation_layout = html.Div(
    [app_shell, _home.layout, _data_browser.layout, _genome_annotations.layout]
)

# --- Register callbacks AFTER layout defined ---
import callbacks.global_filters          # noqa: F401
import callbacks.data_browser_callbacks  # noqa: F401
import callbacks.genome_annotations_callbacks  # noqa: F401
import callbacks.ui_badges  # noqa: F401
import callbacks.home_kpis  # noqa: F401

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
