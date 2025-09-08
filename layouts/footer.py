# layouts/footer.py
from dash import html

def get_footer() -> html.Footer:
    return html.Footer(
        [
            # Section A — Contact & Feedback (top row)
            html.Div(
                [
                    html.H2("Contact & Feedback", className="home-section-title"),
                    html.Span("Questions or ideas? "),
                    html.A("Contact us", href="mailto:juann@ebi.ac.uk"),
                    html.Span(" • "),
                    html.A(
                        "Take our feedback survey",
                        href="https://docs.google.com/forms/d/1vZI2oT06ehqyheihsfVEL9Tnmz5RjxVUjLBsrqEpJX8/edit",
                        target="_blank",
                        rel="noopener",
                    ),
                ],
                className="footer-left",  # reuse existing styles
            ),

            # Section B — Closing banner (full-width, below)
            html.Div(
                html.Img(
                    src="/assets/ARISE-EU_info.png",
                    alt="Project partners",
                    className="footer-banner-img",
                ),
                className="footer-banner",
            ),
        ],
        className="site-footer",
    )
