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

        self.convolution_layers = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(10, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(46656, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 2 * 68),
            nn.Dropout()
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


if __name__ == '__main__':
    import torch
    # instantiate and print your Net
    net = Net()
    print(net)

    t = torch.rand([20, 1, 224, 224])
    print(t.shape)
    print(net(t).shape)

    print(int(10/3 * 16 * 224 / 3 * 224 / 4 / 9))
    print(46656/16)