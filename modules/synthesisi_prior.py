import torch
from torch import nn


class SynthesisPrior(nn.Module):
    def __init__(self, num_channels=192):
        super(SynthesisPrior, self).__init__()
        self.num_channels = num_channels
        num_channels_out = num_channels * 2

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=288, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=288, out_channels=num_channels_out, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), output_padding=(0, 0))
        )

    def forward(self, input_):
        return self.layer(input_)
