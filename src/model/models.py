import torch.nn as nn


class Net4Conv(nn.Module):

    def __init__(self):
        super(Net4Conv, self).__init__()

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
            nn.Dropout(0.3),

            # Fourth Layer: 256 feature maps, 3x3 filter
            # (25 - 3 + 1)/2 = 23/2 = 11.5 -> 11
            # Output: [n, 256, 11, 11]
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 11 * 11, 1028),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1028, 2 * 68),
            nn.Dropout()
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


class Net4ConvV2(nn.Module):

    def __init__(self):
        super(Net4ConvV2, self).__init__()

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
            nn.Dropout(0.3),

            # Fourth Layer: 256 feature maps, 3x3 filter
            # (25 - 3 + 1)/2 = 23/2 = 11.5 -> 11
            # Output: [n, 256, 11, 11]
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 11 * 11, 2 * 68),
            nn.Dropout()
            # nn.Linear(128 * 11 * 11, 1024),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(1024, 2 * 68),
            # nn.Dropout()
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


class Net2Conv(nn.Module):

    def __init__(self):
        super(Net2Conv, self).__init__()

        self.convolution_layers = nn.Sequential(

            # First Layer: 16 feature maps, 5x5 filter
            # (224 - 5 + 1)/2 = 220/2 = 110
            # Output: [n, 16, 110, 110]
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Second Layer: 32 feature maps, 3x3 filter
            # (110 - 3 + 1)/2 = 108/2 = 54
            # Output: [n, 32, 52, 52]
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 54 * 54, 2 * 68),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(2056, 2 * 68),
            nn.Dropout()
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


class Net3Conv(nn.Module):

    def __init__(self):
        super(Net3Conv, self).__init__()

        self.convolution_layers = nn.Sequential(

            # First Layer: 16 feature maps, 5x5 filter
            # (224 - 7 + 1)/2 = 218/2 = 109
            # Output: [n, 16, 109, 109]
            nn.Conv2d(1, 16, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Second Layer: 32 feature maps, 5x5 filter
            # (109 - 3 + 1)/2 = 105/2 = 52.5 -> 52
            # Output: [n, 32, 52, 52]
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Third Layer: 64 feature maps, 5x5 filter
            # (52 - 3 + 1)/2 = 50/2 = 25
            # Output: [n, 32, 52, 52]
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 25 * 25, 2 * 68),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(2056, 2 * 68),
            nn.Dropout()
        )

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


class Net1Conv(nn.Module):

    def __init__(self):
        super(Net1Conv, self).__init__()

        self.convolution_layers = nn.Sequential(

            # First Layer: 32 feature maps, 7x7 filter
            # (224 - 7 + 1)/2 = 218/2 = 109
            # Output: [n, 32, 109, 109]
            nn.Conv2d(1, 32, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 109 * 109, 2056),
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

    names = []
    parameter = []
    forward_time = []

    import torch
    import pandas as pd

    for model in [Net1Conv, Net2Conv, Net3Conv, Net4Conv, Net4ConvV2]:

        net = model()
        name = net.__class__.__name__
        names.append(name)
        start = pd.Timestamp.now()
        t = torch.rand([20, 1, 224, 224])
        assert net(t).shape == (20, 68, 2)
        n_param = sum(p.numel() for p in net.parameters())
        parameter.append(int(n_param / 1e6))
        time_delta = pd.Timedelta(pd.Timestamp.now() - start).microseconds / 1000
        forward_time.append(time_delta)

    df = pd.DataFrame({'Name': names, '# Parameter [Mio]': parameter, 'Forward Time [ms]': forward_time})
    print(df)
