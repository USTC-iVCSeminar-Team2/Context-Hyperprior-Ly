import torch
from torch import nn
import torch.nn.functional as F

class FactorizedModelUnit(nn.Module):

    node_nums = [1, 3, 3, 3, 1]

    def __init__(self, idx = 0, num_channel=128):
        super(FactorizedModelUnit, self).__init__()
        assert idx in range(4), "Wrong unit num"
        self.idx = idx
        d_k = self.node_nums[idx]
        r_k = self.node_nums[idx + 1]
        self.num_channel = num_channel

        self.H = nn.Parameter(torch.nn.init.normal_(tensor=torch.Tensor(self.num_channel, r_k, d_k), mean=0.0, std=0.1))
        self.b = nn.Parameter(torch.nn.init.uniform_(tensor=torch.Tensor(self.num_channel, r_k, 1), a=-0.1, b=0.1))
        self.a = nn.Parameter(
            torch.nn.init.zeros_(tensor=torch.Tensor(self.num_channel, r_k, 1))) if self.idx < 3 else None

    def forward(self, input_):
        H = F.softplus(self.H)
        x = torch.matmul(H, input_) + self.b
        if self.idx < 3:
            a = torch.tanh(self.a)
            output = x + a * torch.tanh(x)
        else:
            output = torch.sigmoid(x)
        return output


class FactorizedModel(nn.Module):
    def __init__(self, num_channel=128, K=4):
        super(FactorizedModel, self).__init__()
        self.num_channel = num_channel
        self.units = nn.ModuleList()
        for i in range(K):
            self.units.append(FactorizedModelUnit(idx=i, num_channel=self.num_channel))

    def forward(self, input_: torch.Tensor):
        B, C, H, W = input_.shape
        assert C == self.num_channel, "Input channel num not fit"
        x = input_.permute((1, 0, 2, 3))
        x = x.reshape(C, 1, -1)
        for unit in self.units:
            x = unit(x)
        output = x.reshape(C, B, H, W).permute((1, 0, 2, 3))
        return output

    def likelihood(self, input_):
        likelihood_ = self.forward(input_ + 0.5) - self.forward(input_ - 0.5) + 1e-6
        return likelihood_


if __name__ == '__main__':
    factorized_model_unit_1 = FactorizedModelUnit(idx=0, num_channel=128)
    factorized_model_unit_2 = FactorizedModelUnit(idx=3, num_channel=128)
    factorized_model = FactorizedModel(num_channel=128)

    input_ = torch.randn(size=(4, 128, 16, 16))
    input_ = input_.permute((1, 0, 2, 3))
    input_ = input_.reshape((128, 1, -1))
    output = factorized_model_unit_1(input_)
    output = factorized_model_unit_2(output)
    print(output.shape)

    input__ = torch.randn(size=(4, 128, 16, 16), requires_grad=True)
    output = factorized_model(input__)
    output = factorized_model.likelihood(input__)
    print(output.shape)



    inout_raw = torch.randn(size=(4, 128, 16, 16))
    loss_ = F.mse_loss(output, inout_raw)
    loss_.backward()
    _ = 0


