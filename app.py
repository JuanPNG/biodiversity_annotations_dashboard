"""
Dash application entry point.

This file assembles the dashboard shell:
- creates the Dash app with Dash Pages enabled,
- exposes the Flask server for deployment,
- installs the shared global filter store,
- mounts the navbar, active page container, and footer,
- imports page modules for validation,
- imports callback modules so Dash registers their callbacks.

Most page-specific UI lives in pages/.
Most page-specific behavior lives in callbacks/.
Shared data access lives in utils/parquet_io.py.
"""


from dash import Dash, html, dcc
import dash
from layouts.navbar import get_navbar
from layouts.footer import get_footer
import os


# Dash Pages lets each module in pages/ register its own route.
# suppress_callback_exceptions=True is needed because callbacks target
# components that may only exist on specific pages.
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Exploring Genome Annotations in Ecological Context",
)

try:
    server = app.server
except NameError:
    raise RuntimeError("Expected Dash 'app' to exist and expose 'server'.")

# --- Health check for Cloud Run / load balancers ---
@server.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return "ok", 200, {"Content-Type": "text/plain"}

# --- Main shell ---
# The app shell is present on every route.
# global-filters is the cross-page state contract: navbar callbacks write it,
# and page callbacks read it to query/filter the data consistently.
app_shell = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="global-filters", storage_type="memory"),
        get_navbar(),
        dash.page_container,
        get_footer(),
    ],
    className="app-root",
    id="app-root",
    **{"data-theme": "embl-light"},  # Set to "dark" to restore the original theme. embl-light
)

app.layout = app_shell

# --- Validation layout: include page layouts so callback IDs are known ---
from pages import home as _home
from pages import data_browser as _data_browser
from pages import genome_annotations as _genome_annotations
from pages import biotype_environment as _biotype_environment
from pages import genome_metrics_environment as _genome_metrics_environment

# Validation layout includes all page layouts so Dash can validate callback IDs
# even when a component is not present on the currently visible route.
app.validation_layout = html.Div(
    [
        app_shell, _home.layout,
        _data_browser.layout,
        _genome_annotations.layout,
        _biotype_environment.layout(),
        _genome_metrics_environment.layout(),
    ]
)

# --- Register callbacks AFTER layout defined ---
# Import callback modules after layouts exist.
# In Dash, importing a module with @callback decorators registers those callbacks.
import callbacks.global_filters          # noqa: F401
import callbacks.data_browser_callbacks  # noqa: F401
import callbacks.genome_annotations_callbacks  # noqa: F401
import callbacks.ui_badges  # noqa: F401
import callbacks.home_kpis  # noqa: F401
import callbacks.biotype_environment_callbacks  # noqa: F401
import callbacks.genome_metrics_environment_callbacks  # noqa: F401

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=True, host="0.0.0.0", port=port)
