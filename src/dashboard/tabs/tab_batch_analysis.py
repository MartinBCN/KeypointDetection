
from dash.dependencies import Output, Input
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dashboard.maindash import app, data_loader, model
import dash_html_components as html
import dash_core_components as dcc


buttons = html.Div(

                    [
                        html.Div(
                            html.Button('Next Batch', id='next-batch', n_clicks=1),
                            style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        ),
                        html.Div(
                            html.Button('Toggle Image', id='toggle-image', n_clicks=1),
                            style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        ),
                        html.Div(
                            html.Button('Toggle Ground Truth', id='toggle-truth', n_clicks=1),
                            style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        ),
                        html.Div(
                            html.Button('Toggle Prediction', id='toggle-prediction', n_clicks=1),
                            style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        )
                    ],
                    style={'width': '55%', 'display': 'inline-block'},
                    className="pretty_container"
            )


def build_batch_analysis():
    plot = html.Div(
            [
                html.Div(
                    dcc.Graph(id='batch_analysis',
                              config={'displayModeBar': False}
                              ),
                    style={'width': '95%', 'display': 'inline-block'},
                    className="pretty_container"
                ),
                html.Div(
                    [
                        html.Div(style={'width': '25%', 'display': 'inline-block'}),
                        buttons,
                        html.Div(style={'width': '25%', 'display': 'inline-block'}),
                    ]
                )
            ]
        )
    return plot


def plot_single_image(i: int, image: np.array, keypoints: np.array, prediction: np.array, fig,
                      show_image: bool, show_truth: bool, show_prediction: bool) -> None:

    print('show_image', show_image)
    print('show_truth', show_truth)
    print('show_prediction', show_prediction)

    if show_image:
        fig.add_trace(
            px.imshow(image[0, :, :], color_continuous_scale='gray').data[0],
            row=1, col=i+1
        )

    if show_truth:
        fig.add_trace(
            go.Scatter(
                    x=keypoints[:, 0],
                    y=keypoints[:, 1],
                    name='Ground Truth',
                    line={'color': px.colors.qualitative.Plotly[0]},
                    mode='markers',
                    showlegend=False,
                ),
            row=1, col=i+1
        )

    if show_prediction:
        fig.add_trace(
            go.Scatter(
                    x=prediction[:, 0],
                    y=prediction[:, 1],
                    name='Ground Truth',
                    line={'color': px.colors.qualitative.Plotly[1]},
                    mode='markers',
                    showlegend=False,
                ),
            row=1, col=i+1
        )


@app.callback(
    Output('batch_analysis', 'figure'),
    [Input('next-batch', 'n_clicks'),
     Input('toggle-image', 'n_clicks'),
     Input('toggle-truth', 'n_clicks'),
     Input('toggle-prediction', 'n_clicks')]
)
def update_batch_plot(next_batch: int, toggle_image_clicks: int,
                      toggle_truth_clicks: int, toggle_prediction: int) -> dict:
    return plot_batch(next_batch, toggle_image_clicks, toggle_truth_clicks, toggle_prediction)


def plot_batch(next_batch: int, toggle_image_clicks: int,  toggle_truth_clicks: int, toggle_prediction: int):

    show_image = (toggle_image_clicks % 2) == 1
    show_truth = (toggle_truth_clicks % 2) == 1
    show_prediction = (toggle_prediction % 2) == 1

    # Batch
    batch = next(iter(data_loader))
    image = batch['image']
    keypoints = batch['keypoints']
    prediction = model.model(image)

    # Numpy
    image = image.cpu().numpy()
    kp = keypoints.detach().cpu().numpy()
    kp = kp * 50 + 100
    pred = prediction.detach().cpu().numpy()
    pred = pred * 50 + 100

    n = image.shape[0]

    fig = make_subplots(rows=1, cols=n)
    for i in range(n):
        plot_single_image(i, image[i, :, :, :], kp[i, :, :], pred[i, :, :], fig,
                          show_image, show_truth, show_prediction)

    for ax in ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4', 'yaxis5']:
        fig['layout'][ax]['autorange'] = 'reversed'

    return fig
