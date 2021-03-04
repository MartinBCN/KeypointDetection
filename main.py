from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import FacialKeypointsDataset
from src.data.transforms import Rescale, RandomCrop, Normalize, ToTensor
from src.model.models import Net

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
"""

    Convert the face from RGB to grayscale
    Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    Reshape the numpy image into a torch image.

"""

# Construct the dataset
face_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                      root_dir='data/training/',
                                      transform=data_transform)

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                      root_dir='data/test/',
                                      transform=data_transform)

# print some stats about the dataset
print('Length of dataset: ', len(face_dataset))


# load training data in batches
batch_size = 10

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

train_loader = DataLoader(face_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

for sample in test_loader:
    net = Net()
    image = sample['image']
    print(image.shape)
    print(sample['keypoints'].shape)
    # prediction = net(image)
    # print(prediction.shape)
    break
