import os

import dash
import flask
import dash_bootstrap_components as dbc

from model.keypoint_detector import KeypointDetector

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

model_path = os.environ.get('MODEL_PATH', '../models')
fn = f'{model_path}/test1.pt'
model = KeypointDetector.load(fn)
