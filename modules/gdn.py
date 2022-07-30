import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class SetMinBoundary(Function):
    """
    Set parameter in GDN to min boundary after each gradient step which is 2^-5 in the paper.
    """

    @staticmethod
    def forward(ctx, input, min_boundary):
        b = torch.ones_like(input) * min_boundary
        ctx.save_for_backward(input, b)
        return torch.max(input, b)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: gradient from previous layer
        :return: grandient
        """
        input, b = ctx.saved_tensors
        passthrough_map = (input >= b) | (grad_output < 0)
        return passthrough_map.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self, num_channel=128, beta_min=1e-6, beta_init=0.1, gamma_min=1e-6, gamma_init=0.1,
                 min_boundary=2**-5, inverse=False):       # N for channle nums
        super(GDN, self).__init__()
        self.num_channel = num_channel
        self.inverse = inverse

        self.reparam_offset = min_boundary ** 2
        self.beta_bound = (beta_min + self.reparam_offset) ** 0.5
        self.gamma_bound = (gamma_min + self.reparam_offset) ** 0.5

        # beta, gamma
        self.beta = nn.Parameter(torch.sqrt(torch.ones(num_channel) * beta_init + self.reparam_offset))
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(num_channel) * gamma_init + self.reparam_offset))

    def forward(self, input_):
        B, C, H, W = input_.shape
        assert self.num_channel == C, "Input channel num not fit"

        gamma_T = self.gamma.transpose(0, 1)
        gamma_p = (self.gamma + gamma_T) / 2

        beta_p = SetMinBoundary.apply(self.beta, self.beta_bound)
        beta = beta_p ** 2 - self.reparam_offset

        gamma_p = SetMinBoundary.apply(gamma_p, self.gamma_bound)
        gamma = gamma_p ** 2 - self.reparam_offset
        gamma = gamma.reshape(self.num_channel, self.num_channel, 1, 1)

        norm = F.conv2d(input=input_**2, weight=gamma, bias=beta)
        norm = torch.sqrt(norm)

        if self.inverse:
            return input_ * norm
        else:
            return input_ / norm

if __name__ == '__main__':
    gdn = GDN()
    gdn_i = GDN(inverse=True)

    input_ = torch.randn(size=(4, 128, 16, 16), requires_grad=True)
    output = gdn(input_)
    input_reco = gdn_i(output)

    input_std = torch.randn(size=input_.shape)
    loss = F.mse_loss(input_reco, input_std)
    loss.backward()

    _ = 0