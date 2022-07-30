import torch
from torch import nn


class EntropyParameters(nn.Module):
    def __init__(self, num_channels=192*2*2):
        super(EntropyParameters, self).__init__()
        self.num_channels = num_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=640, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=640, out_channels=512, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=384, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, input_):
        return self.layer(input_)


