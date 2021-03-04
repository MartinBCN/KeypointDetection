from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import FacialKeypointsDataset
from src.data.loader import get_data_loader
from src.data.transforms import Rescale, RandomCrop, Normalize, ToTensor
from src.model.models import Net


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

loader = get_data_loader(data_type='test', batch_size=10, data_transform=data_transform)

for sample in loader:
    net = Net()
    image = sample['image']
    print(image.shape)
    print(sample['keypoints'].shape)
    prediction = net(image)
    print(prediction.shape)
    break
