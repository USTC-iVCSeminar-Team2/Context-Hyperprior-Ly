import torch
from torch import nn
from .gdn import GDN


class Synthesis(nn.Module):
    def __init__(self, num_channels=192):
        super(Synthesis, self).__init__()
        self.num_channels = num_channels

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=num_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=num_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=num_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=3, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1))
        )

    def forward(self, input_):
        return self.layer(input_)
