import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class ToKeyPoints(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1, 2)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(10, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            Flatten(),
            nn.Linear(46656, 2 * 68),

            ToKeyPoints()
        )

    def forward(self, x):

        return self.main(x)


if __name__ == '__main__':
    import torch
    # instantiate and print your Net
    net = Net()
    print(net)

    t = torch.rand([20, 1, 224, 224])
    print(t.shape)
    print(net(t).shape)