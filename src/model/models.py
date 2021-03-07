import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.convolution_layers = nn.Sequential(

            # First Layer: 32 feature maps, 7x7 filter
            # (224 - 7 + 1)/2 = 218/2 = 109
            # Output: [n, 32, 109, 109]
            nn.Conv2d(1, 32, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Second Layer: 64 feature maps, 5x5 filter
            # (109 - 5 + 1)/2 = 105/2 = 52.5 -> 52
            # Output: [n, 64, 52, 52]
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Third Layer: 128 feature maps, 3x3 filter
            # (52 - 3 + 1)/2 = 50/2 = 25
            # Output: [n, 128, 25, 25]
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 25 * 25, 2056),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2056, 2 * 68),
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
    print(128 * 25 * 25)

    t = torch.rand([20, 1, 224, 224])
    print(t.shape)
    print(net(t).shape)

