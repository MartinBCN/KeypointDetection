import base64
import json
import os
from pathlib import Path
import random

import flask
from PIL import Image
from dash.dependencies import Output, Input

from dashboard.maindash import app
import dash_html_components as html


def build_batch_analysis():

    return html.Div(children=[])
