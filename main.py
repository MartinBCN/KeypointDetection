import os

from torch import optim
from torch.nn import MSELoss, L1Loss
from torchvision import transforms

from src.data.loader import get_data_loader
from src.data.transforms import Rescale, RandomCrop, Normalize, ToTensor
from src.model.keypoint_detector import KeypointDetector

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
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

loader = dict()
loader['validation'] = get_data_loader(data_type='test', batch_size=10, data_transform=data_transform)
loader['train'] = get_data_loader(data_type='training', batch_size=10, data_transform=data_transform)

model = KeypointDetector()
model.set_criterion('l1')
model.set_optimizer('sgd', dict(lr=0.001, momentum=0.9))

# model.train(loader, 2)

model_path = os.environ.get('MODEL_PATH', 'models')
fn = f'{model_path}/test1.pt'
model.save(fn)

loaded_model = model.load(fn)

for sample in loader['validation']:
    output = loaded_model.model(sample['image'])
    print(output.shape)
    break
