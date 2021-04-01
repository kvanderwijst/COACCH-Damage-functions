import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

from .fitfunctions import GeneralFitFunction

# Bugfix for Plotly default export size
import plotly.io as pio

pio.kaleido.scope.default_width = None
pio.kaleido.scope.default_height = None


def create_single_plot(
    regional_data, x_param, fit_fct: GeneralFitFunction, regional_coeffs
):

    x = np.linspace(
        min(0, 1.05 * regional_data[x_param].min()), 1.05 * regional_data[x_param].max()
    )
    y_best = fit_fct.fct(x, *regional_coeffs[fit_fct.param_names])

    # Add markers
    fig = px.scatter(regional_data, x=x_param, y="Value", render_mode="svg")

    # Add background (95 conf. interval)
    a_low = regional_coeffs["a (q=0.025)"]
    a_high = regional_coeffs["a (q=0.975)"]
    fig.add_scatter(x=x, y=a_low * y_best, showlegend=False, line_width=0)
    fig.add_scatter(
        x=x,
        y=a_high * y_best,
        name="95th p. conf int",
        line_width=0,
        fillcolor="rgba(0,0,0,.15)",
        fill="tonexty",
    )

    # Add other quantiles
    for q in [0.05, 0.16, 0.5, 0.84, 0.95]:
        a = regional_coeffs[f"a (q={q})"]
        color = "rgb({v},{v},{v})".format(v=255 * 2 * abs(0.5 - q))
        fig.add_scatter(x=x, y=a * y_best, name=f"q={q}", line_color=color)

    fig.add_scatter(x=x, y=y_best, line_color="mediumvioletred", name="best fit")

    return fig.update_traces(marker_size=4)


def create_combined_plot(data, region, all_coeffs, max_cols=2, with_histogram=True):
    regional_data = data[data["Region"] == region]

    nrows = int(np.ceil(len(all_coeffs) / max_cols))
    ncols = min(2, len(all_coeffs))

    subplot_titles = []
    for coeffs in all_coeffs:
        subplot_titles.extend(
            [
                "{}<br><i>a</i> * ( {} )".format(
                    coeffs["fit_fct"].__name__,
                    coeffs["fit_fct"].formula.format(x=coeffs["x_param"]),
                ),
            ]
            + (["Hist. of <i>a</i>"] if with_histogram else [])
        )

    fig = make_subplots(
        nrows,
        2 * ncols if with_histogram else ncols,
        horizontal_spacing=0.03,
        subplot_titles=subplot_titles,
        column_widths=[3 / (4 * ncols), 1 / (4 * ncols)] * ncols
        if with_histogram
        else None,
    )

    for i, coeffs in enumerate(all_coeffs):
        col = 2 * (i % ncols) + 1 if with_histogram else i % ncols + 1
        row = i // ncols + 1
        subfig = create_single_plot(
            regional_data,
            coeffs["x_param"],
            coeffs["fit_fct"],
            coeffs["values"].loc[region],
        )
        for trace in subfig.data:
            fig.add_trace(
                trace.update(showlegend=i == 0 and trace.showlegend), row=row, col=col
            )
        fig.update_xaxes(title=coeffs["x_param"], col=col, row=row)
        fig.update_yaxes(ticksuffix="%", matches="y1", row=row, col=col)
        if with_histogram:
            # Add histogram
            fig.add_histogram(
                x=coeffs["ratios"][region],
                showlegend=False,
                marker_color=px.colors.qualitative.Plotly[0],
                row=row,
                col=col + 1,
            )
            fig.update_xaxes(title="<i>a</i>", col=col + 1, row=row)

    # Update layout
    fig.update_layout(
        width=1100,
        legend_traceorder="reversed",
        title=f"Region: <b>{region}</b>",
        height=100 + 350 * nrows,
    )

    return fig
