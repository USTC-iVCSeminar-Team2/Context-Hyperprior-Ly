import torch
from torch import nn
import torch.nn.functional as F
from modules import *
from time import time


class ContextHyperPrior(nn.Module):
    def __init__(self, a, h, rank, num_channels=192):
        super(ContextHyperPrior, self).__init__()
        self.a = a
        self.h = h
        self.num_channels = num_channels
        self.device = torch.device('cuda:{}'.format(rank)) if rank != 'cpu' else torch.device('cpu')
        # Encoders and decoders
        self.g_a = Analysis(num_channels=self.num_channels)
        self.g_s = Synthesis(num_channels=self.num_channels)
        self.h_a = AnalysisPrior(num_channels=self.num_channels)
        self.h_s = SynthesisPrior(num_channels=self.num_channels)
        # Context model and entropy parameters model
        self.context_model = ContextModel(num_channels=self.num_channels)
        self.entropy_parameters = EntropyParameters(num_channels=self.num_channels*2*2)
        # Entropy model
        self.factorized_model = FactorizedModel(num_channel=self.num_channels, K=4)
        self.gaussian_mixture_model = GaussianMixtureModel()
        # Entropy codec
        self.entropy_coder_factorized = EntropyCoder(self.factorized_model)
        self.entropy_coder_gaussian_mixture = EntropyCoderGaussianMixture(self.gaussian_mixture_model)

    def quantize(self, input_, is_tain=True):
        if is_tain:
            uniform_noise = torch.nn.init.uniform_(torch.zeros_like(input_), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            return input_ + uniform_noise
        else:
            return torch.round(input_)

    def forward(self, input_):
        x = input_
        # latent variable y and hyperprior z
        y = self.g_a(x)
        y_hat = self.quantize(y, is_tain=True)
        z = self.h_a(y)
        z_hat = self.quantize(z, is_tain=True)
        # hyper \psi and autoregressive prediction \phy
        params_hyper = self.h_s(z_hat)
        params_ctx = self.context_model(y_hat)
        # \miu and \sigma for gaussian
        params_entropy = torch.cat((params_hyper, params_ctx), dim=1)
        gaussian = self.entropy_parameters(params_entropy)
        mean, scale = torch.chunk(gaussian, chunks=2, dim=1)
        # bpp estimation using factorized model and gaussian model
        bits_z = torch.sum(torch.clamp(-torch.log2(self.factorized_model.likelihood(z_hat)), min=0, max=50))
        bpp_z = bits_z / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        bits_y = torch.sum(torch.clamp(-torch.log2(self.gaussian_mixture_model.likelihood(y_hat, mean, scale)), min=0, max=50))
        bpp_y = bits_y / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        # MSE disstortion
        x_hat = torch.clamp(self.g_s(y_hat), min=0, max=1)
        disstortion = torch.mean((x - x_hat) ** 2)
        # loss = \lambda * 255^2 * D + R
        loss = (bpp_y + bpp_z) + self.a.lambda_ * (255 ** 2) * disstortion
        return loss, bpp_y, bpp_z, disstortion, x_hat

    def inference(self, input_):
        # ---------------- Encoding ----------------
        time_enc_start = time()
        x = input_
        # latent variable y and hyperprior z
        y = self.g_a(x)
        y_hat = self.quantize(y, is_tain=False)
        z = self.h_a(y)
        z_hat = self.quantize(z, is_tain=False)
        # hyper \psi and autoregressive prediction \phy
        params_hyper = self.h_s(z_hat)
        params_ctx = self.context_model(y_hat)
        # \miu and \sigma for gaussian
        params_entropy = torch.cat((params_hyper, params_ctx), dim=1)
        gaussian = self.entropy_parameters(params_entropy)
        mean, scale = torch.chunk(gaussian, chunks=2, dim=1)
        # entropy encode
        stream_z, side_info_z = self.entropy_coder_factorized.compress(z_hat)
        stream_y, side_info_y = self.entropy_coder_gaussian_mixture.compress(y_hat, mean, scale)
        time_enc_end = time()

        # ---------------- Decoding ----------------
        time_dec_start = time()
        # decode z_hat from stream
        z_hat_dec = self.entropy_coder_factorized.decompress(stream_z, side_info_z, self.device)
        assert torch.equal(z_hat, z_hat_dec), "Entropy code decode for z_hat not consistent !"
        # hyper \psi
        params_hyper_dec = self.h_s(z_hat_dec)
        assert torch.equal(params_hyper, params_hyper_dec)
        # zero y_hat at start
        y_hat_dec = torch.zeros_like(y_hat).to(self.device)
        height, width = y_hat_dec.shape[2:]
        # Decoding iteration
        for h in range(height):
            for w in range(width):
                # autoregressive prediction \phy
                params_ctx_dec = self.context_model(y_hat_dec)
                # # \miu and \sigma for gaussian
                params_entropy_dec = torch.cat((params_hyper_dec, params_ctx_dec), dim=1)
                # assert (torch.equal(params_entropy[:, :, h: h + 1, w: w + 1], params_entropy_dec[:, :, h: h + 1, w: w + 1]))
                gaussian_dec = self.entropy_parameters(params_entropy_dec)
                # assert (torch.equal(gaussian[:, :, h: h + 1, w: w + 1], gaussian_dec[:, :, h: h + 1, w: w + 1]))
                mean_dec, scale_dec = torch.chunk(gaussian_dec, chunks=2, dim=1)
                # assert (torch.equal(mean[:, :, h, w], mean_dec[:, :, h, w]))
                # assert (torch.equal(scale[:, :, h, w], scale_dec[:, :, h, w]))
                temp = self.entropy_coder_gaussian_mixture.decompress(stream_y, side_info_y, mean_dec, scale_dec, self.device)
                y_hat_dec[:, :, h, w] = temp[:, :, h, w]
                assert (torch.equal(y_hat[:, :, h, w], y_hat_dec[:, :, h, w]))

        assert torch.equal(y_hat, y_hat_dec), "Entropy code decode for y_hat not consistent !"

        x_hat = torch.clamp(self.g_s(y_hat_dec), min=0, max=1)
        time_dec_end = time()
        # print("Enc. time, dec. time: {:.4f}, {:.4f}".format((time_enc_end - time_enc_start), (time_dec_end - time_dec_start)))

        bpp_y = len(stream_y) * 8 / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        bpp_z = len(stream_z) * 8 / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        return x_hat, bpp_y, bpp_z, (time_enc_end - time_enc_start), (time_dec_end - time_dec_start)


if __name__ == '__main__':
    model = ContextHyperPrior()
    _ = model.modules()
    _ = 0
