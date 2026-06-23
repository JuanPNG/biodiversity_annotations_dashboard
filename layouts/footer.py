"""
Shared footer layout.

The footer is mounted once in app.py and appears below every Dash Pages route.
It contains contact, feedback, and project partner information.
"""
from dash import html


# Build the footer component used by the app shell.
def get_footer() -> html.Footer:
    return html.Footer(
        [
            html.Div(
                [
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
                        className="footer-left",
                    ),
                    html.Div(
                        [
                            html.Img(
                                src="/assets/AriseLogo.png",
                                alt="ARISE logo",
                                className="footer-arise-logo",
                            ),
                            html.Img(
                                src="/assets/ARISE-EU_info.png",
                                alt="Project partners",
                                className="footer-banner-img",
                            ),
                        ],
                        className="footer-banner",
                    ),
                ],
                className="footer-content",
            ),
        ],
        className="site-footer",
    )
