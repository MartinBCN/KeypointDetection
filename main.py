import os

from data.loader import get_data_loader
from data.transforms import Rescale, RandomCrop, Normalize, ToTensor
from data.visualise import visualise_batch
from model.keypoint_detector import KeypointDetector

from torchvision import transforms
import torch
torch.manual_seed(42)

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

"""

    Convert the face from RGB to grayscale
    Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    Reshape the numpy image into a torch image.

"""

# --- Data ---
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

fraction = 0.05
batch_size = 5
loader = {data_type: get_data_loader(data_type=data_type, batch_size=batch_size, data_transform=data_transform,
                                     fraction=fraction) for data_type in ['train', 'validation']}

# --- Setup Model ---
model = KeypointDetector()
model.set_criterion('l1')
model.set_optimizer('sgd', dict(lr=0.001, momentum=0.9))

# --- Training ---
model.train(loader, 2)

# --- Plot Loss ---
figure_dir = os.environ.get('FIG_PATH', 'figures')
fn = f'{figure_dir}/test.png'
model.plot(fn)

# --- Store Model ---
model_path = os.environ.get('MODEL_PATH', 'models')
fn = f'{model_path}/test1.pt'
model.save(fn)

sample = next(iter(loader['validation']))
image = sample['image']
keypoints = sample['keypoints']
prediction = model.model(image)
visualise_batch(image, keypoints, prediction)
