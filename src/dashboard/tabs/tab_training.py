from plotly.subplots import make_subplots
import plotly.express as px
import dash_html_components as html
import plotly.graph_objects as go

import dash_core_components as dcc
from dashboard.maindash import model

PHASES = ['train', 'validation']


def plot_stats() -> dict:

    fig = make_subplots(rows=2, cols=2)

    make_traces(field='batch_loss', row=1, col=1, fig=fig)
    make_traces(field='epoch_loss', row=1, col=2, fig=fig)
    make_traces(field='batch_accuracy', row=2, col=1, fig=fig)
    make_traces(field='epoch_accuracy', row=2, col=2, fig=fig)

    return fig


def make_traces(field: str, row: int, col: int, fig):

    for i, phase in enumerate(PHASES):
        y = model.training_log[phase][field]
        x = list(range(len(y)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=phase,
                line={'color': px.colors.qualitative.Plotly[i]},
                showlegend=(field == 'batch_loss')
            ),
            row=row, col=col
        )


def build_training_stats():

    plot = html.Div(
            [
                html.Div(
                    dcc.Graph(id='batch_loss',
                              config={'displayModeBar': False},
                              figure=plot_stats()
                              ),
                    style={'width': '95%', 'display': 'inline-block'},
                    className="pretty_container"
                )

            ]
        )

    return plot

