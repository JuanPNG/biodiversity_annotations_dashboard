import plotly.graph_objects as go


EMBL_PLOTLY_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#332288",  # indigo
    "#88CCEE",  # light blue
    "#AA4499",  # purple
    "#117733",  # dark green
    "#DDCC77",  # muted yellow
    "#888888",  # grey
]


def apply_embl_theme(fig: go.Figure) -> go.Figure:
    """Apply the EMBL-EBI light visual style to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": "IBM Plex Sans, Helvetica, Arial, sans-serif",
            "color": "#373A36",
            "size": 13,
        },
        colorway=EMBL_PLOTLY_COLORS,
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#D0D0CE",
            "font": {"color": "#373A36"},
        },
        legend={
            "font": {"color": "#373A36"},
        },
    )
    fig.update_xaxes(
        gridcolor="#E6E7E3",
        zerolinecolor="#D0D0CE",
        linecolor="#D0D0CE",
        tickcolor="#707372",
        title_font={"color": "#373A36"},
        tickfont={"color": "#707372"},
    )
    fig.update_yaxes(
        gridcolor="#E6E7E3",
        zerolinecolor="#D0D0CE",
        linecolor="#D0D0CE",
        tickcolor="#707372",
        title_font={"color": "#373A36"},
        tickfont={"color": "#707372"},
    )
    return fig