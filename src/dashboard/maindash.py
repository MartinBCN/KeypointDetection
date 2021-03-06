import os

import dash
import flask
import dash_bootstrap_components as dbc

from torchvision import transforms

from data.loader import get_data_loader
from data.transforms import Rescale, RandomCrop, Normalize, ToTensor
from model.keypoint_detector import KeypointDetector


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

model_path = os.environ.get('MODEL_PATH', '../models')
fn = f'{model_path}/test1.pt'
model = KeypointDetector.load(fn)

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
os.environ['DATA_PATH'] = '../data'
data_loader = get_data_loader(data_type='test', batch_size=5, data_transform=data_transform)