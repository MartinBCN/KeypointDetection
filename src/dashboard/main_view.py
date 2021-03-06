
import dash_html_components as html
import dash_core_components as dcc

from dashboard.maindash import app
from dashboard.tabs.tab_training import build_training_stats
from dashboard.tabs.tab_batch_analysis import build_batch_analysis
from dashboard.tabs.tab_upload_image import build_upload_image

logo = html.Img(
    src=app.get_asset_url("dash-logo.png"),
    id="plotly-image",
    style={
        "height": "60px",
        "width": "auto",
        "margin-bottom": "25px",
    },
)


def build_banner():
    return html.Div(
            [
                html.Div(
                    [logo],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Facial Keypoint Detection",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Analysis", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-third column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Source", id="learn-more-button"),
                            href="https://github.com/MartinBCN/KeypointDetection",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        )


def build_tabs():
    tab_training = dcc.Tab(
                        id="tab-training",
                        label="Training Stats",
                        value="tab_training",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_training_stats()
    )

    tab_batch_analysis = dcc.Tab(
                        id="tab-batch-analysis",
                        label="Batch Analysis",
                        value="tab_batch_analysis",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_batch_analysis()
                    )

    tab_upload_image = dcc.Tab(
                        id="tab-upload-image",
                        label="Upload Image",
                        value="tab_upload_image",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_upload_image()
                    )

    tabs = html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="tabs-keypoint",
                value="tab_batch_analysis",  # Default Choice
                className="custom-tabs",
                children=[tab_training, tab_batch_analysis, tab_upload_image],
            )
        ],
    )

    return tabs


def main_layout():
    return html.Div(children=[
                html.Div(id="output-clientside"),
                build_banner(),
                build_tabs()
            ],
                id="mainContainer",
                style={"display": "flex", "flex-direction": "column"})
