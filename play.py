import torch
import torch.nn.functional as F

from time import time
import random

from models.model import ContextHyperPrior
from modules import MaskedConv2d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def test_masked_conv():
    # Once
    masked_conv2d = MaskedConv2d(in_channels=192, out_channels=384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    weight_ = torch.randn(size=(384, 192, 5, 5))
    masked_conv2d.weight.data = weight_
    y_hat = torch.randn(size=(1, 192, 16, 16))
    time_once_start = time()
    output_once = masked_conv2d(y_hat)
    time_once_end = time()
    time_once = time_once_end - time_once_start

    # Iteration
    y_hat_pad = F.pad(y_hat, (2, 2, 2, 2))
    masked_weight_iter = weight_ * masked_conv2d.mask
    height, width = y_hat.shape[2:]
    params_ctx = torch.zeros(size=(1, 384, 16, 16))
    time_iter_start = time()
    for h in range(height):
        for w in range(width):
            y_hat_crop = y_hat_pad[:, :, h: h + 5, w: w + 5]
            params_ctx[:, :, h: h + 1, w: w + 1] = F.conv2d(y_hat_crop, masked_weight_iter, masked_conv2d.bias)
    time_iter_end = time()
    time_iter = time_iter_end - time_iter_start

    _ = torch.equal(output_once, params_ctx)
    _ = 0


if __name__ == '__main__':
    # masked_conv = MaskedConv2d(in_channels=192, out_channels=192, kernel_size=(5, 5), padding=(1, 1))
    # _ = masked_conv.mask
    # test_masked_conv()

    arg_dict = {'lambda_': 0.0483}
    a = AttrDict(**arg_dict)

    # analysis_prior_net = AnalysisPrior()
    # synthesis_prior_net = SynthesisPrior()
    compressor = ContextHyperPrior(a=a, h='', rank='cpu', num_channels=192)

    x = torch.randn(size=(1, 3, 256, 256))
    compressor.inference(x)

    loss, bpp_y, bpp_z, disstortion, x_hat = compressor(x)
    loss.backward()

    _ = 0
