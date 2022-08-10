import torch
from torch import nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type='A', **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        # type 'A' to mask the center, from PixelCNN
        if mask_type not in ('A', 'B'):
            raise ValueError("Invalid \"mask_type\" value: {}".format(mask_type))
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, h, w = self.mask.size()
        # setting below weights to 0
        self.mask[:, :, (h // 2), (w // 2 + (mask_type == 'B')):] = 0
        self.mask[:, :, (h // 2 + 1):, :] = 0

    def forward(self, input_):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(input_)


class ContextModel(nn.Module):
    def __init__(self, num_channels=192):
        super(ContextModel, self).__init__()
        self.num_channels = num_channels
        num_channels_out = self.num_channels * 2

        self.layer = MaskedConv2d(mask_type='A', in_channels=self.num_channels, out_channels=num_channels_out,
                                  kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, input_):
        return self.layer(input_)



if __name__ == '__main__':
    masked_conv2d = MaskedConv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), mask_type='A')
    context_model = ContextModel()

    x = torch.ones(1, 2, 3, 3)
    masked_conv2d.weight.data[:, :, :, :] = 2
    output = masked_conv2d(x)

    x = torch.randn(size=(4, 192, 16, 16))
    output = context_model(x)
    _ = 0