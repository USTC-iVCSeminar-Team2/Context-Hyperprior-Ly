import torch
from torch import nn
from .gdn import GDN


class Analysis(nn.Module):
    def __init__(self, num_channels=192):
        super(Analysis, self).__init__()
        self.num_channels = num_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            GDN(num_channel=num_channels, inverse=False),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                      padding=(2, 2)),
            GDN(num_channel=num_channels, inverse=False),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                      padding=(2, 2)),
            GDN(num_channel=num_channels, inverse=False),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                      padding=(2, 2))
        )

    def forward(self, input_):
        return self.layer(input_)
