import torch
from torch import nn


class AnalysisPrior(nn.Module):
    def __init__(self, num_channels=192):
        super(AnalysisPrior, self).__init__()
        self.num_channels = num_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(5, 5), stride=(2, 2),
                      padding=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(5, 5), stride=(2, 2),
                      padding=(2, 2))
        )

    def forward(self, input_):
        return self.layer(input_)
