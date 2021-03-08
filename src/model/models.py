from typing import List
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, image_size: int, filter_size: List[int], feature_maps: List[int],
                 second_linear_layer: bool = True):

        super(Net, self).__init__()
        self.image_size = image_size
        self.filter_size = filter_size
        self.features = feature_maps
        self.input_size = [1] + feature_maps[:-1]

        output_size = 224

        conv_layers = []
        for input, features, filter in zip(self.input_size, self.features, self.filter_size):

            output_size = int((output_size - filter + 1) / 2)
            conv_layers.append(nn.Conv2d(input, features, filter))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2),)
            conv_layers.append(nn.Dropout(p=0.3))

        self.convolution_layers = nn.Sequential(*conv_layers)

        if second_linear_layer:

            self.linear_layers = nn.Sequential(

                nn.Linear(output_size**2 * self.features[-1], 1024),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 136)

            )

        else:
            self.linear_layers = nn.Linear(output_size**2 * self.features[-1], 136)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


if __name__ == '__main__':

    import torch
    import pandas as pd
    from torch.optim import Adam

    parameter = []
    forward_time = []
    backward_time = []
    feature_maps = []
    filter_sizes = []
    second_linear = []

    models = []
    models.append({'Filter Sizes': [3], 'Feature Maps': [128], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [3], 'Feature Maps': [128], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3], 'Feature Maps': [32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3], 'Feature Maps': [32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [3, 1], 'Feature Maps': [32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [3, 1], 'Feature Maps': [32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3, 1], 'Feature Maps': [16, 32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3, 1], 'Feature Maps': [16, 32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128, 256],
                   'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128, 256],
                   'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [32, 64, 128, 256, 512],
                   'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [32, 64, 128, 256, 512],
                   'Second Linear Layer': False})


    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    for model in models:

        # --- Instantiate ---
        net = Net(image_size=224, filter_size=model['Filter Sizes'], feature_maps=model['Feature Maps'],
                       second_linear_layer=model['Second Linear Layer'])
        criterion = nn.MSELoss()
        optimizer = Adam(net.parameters())

        n_param = sum(p.numel() for p in net.parameters())
        parameter.append(int(n_param / 1e6))

        # --- Forward Step ---
        t = torch.rand([20, 1, 224, 224])
        start = pd.Timestamp.now()
        output = net(t)
        time_delta = pd.Timedelta(pd.Timestamp.now() - start).microseconds / 1000
        assert output.shape == (20, 68, 2)
        forward_time.append(time_delta)

        # --- Backward Step ---
        start = pd.Timestamp.now()
        loss = criterion(output, torch.rand([20, 68, 2]))
        optimizer.step()
        time_delta = pd.Timedelta(pd.Timestamp.now() - start).microseconds / 1000
        backward_time.append(time_delta)

        feature_maps.append(model['Feature Maps'])
        filter_sizes.append(model['Filter Sizes'])
        second_linear.append(model['Second Linear Layer'])

    df = pd.DataFrame({'Feature Maps': feature_maps, 'Filter Sizes': filter_sizes, '# Parameter [Mio]': parameter,
                       'Second Linear Layer': second_linear,
                       'Forward Time [ms]': forward_time, 'Backward Time [ms]': backward_time})
    print(df)
