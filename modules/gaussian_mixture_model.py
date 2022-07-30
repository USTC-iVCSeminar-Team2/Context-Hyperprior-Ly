import torch
from torch import nn
import torch.nn.functional as F
from modules import SetMinBoundary


class GaussianMixtureModel(nn.Module):
    def __init__(self):
        super(GaussianMixtureModel, self).__init__()

    def standardized_cumulative(self, input_):
        return 1.0 - 0.5 * torch.erfc((2 ** -0.5) * input_)     # c(x) = 1 - c(-x) = 0.5 * erfc(-(2**-0.5) * x)

    def forward(self, input_, mean, scale):
        assert (input_.shape[0:3] == mean.shape[0:3]) or (
                    input_.shape[0:3] == scale.shape[0:3]), "Shape dismatch between y and gaussian mean scale"
        scale = SetMinBoundary.apply(scale, 1e-6)
        input_ = (input_ - mean) / scale
        cumul = self.standardized_cumulative(input_)
        return cumul

    def likelihood(self, input_, mean, scale):
        likelihood_ = self.forward(input_ + 0.5, mean, scale) - self.forward(input_ - 0.5, mean, scale) + 1e-6
        return likelihood_


if __name__ == '__main__':
    gaussian_mixture_model = GaussianMixtureModel()

    x = torch.nn.init.normal_(torch.Tensor(4 ,128, 16, 16), mean=1.5, std=2.0)
    mean = torch.ones(x.shape, dtype=torch.float32) * 1.5
    scale = torch.ones(x.shape, dtype=torch.float32) * 2.0

    output = gaussian_mixture_model(x, mean, scale)
    output_ = gaussian_mixture_model.likelihood(x, mean, scale)
    _ = 0

