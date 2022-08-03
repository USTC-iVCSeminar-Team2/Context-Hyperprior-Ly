import torch
import struct
import os
from modules import FactorizedModel
from modules import GaussianMixtureModel
import torchac

class EntropyCoder():
    '''
    Base class for entropy coding
    '''
    def __init__(self, entropy_model):
        self.entropy_model = entropy_model

    def pmf_to_cdf(self, pmf):
        '''
        :param pmf: the probs of all possible symbols, shape [B, C, 1, L]
        :return: cdf from pmf, shape [B, C, 1, L+1]
        '''
        # Make the sum of pmf equal to 1
        pmf_sum = torch.sum(pmf, dim = 3).reshape(*pmf.shape[0:3], 1)
        pmf_norm = torch.div(pmf, pmf_sum)
        # Add pmf together to get cdf
        cdf = torch.zeros(pmf_norm.shape).to(pmf.device)
        for i in range(0, pmf_norm.shape[-1]):
            cdf[:, :, :, i:] += pmf_norm[:, :, :, i].reshape(*pmf.shape[0:3], 1)
        # Add a beginning 0 for cdf
        cdf = torch.cat((torch.zeros(*pmf.shape[0:3], 1).to(pmf.device), cdf), dim=-1).clamp(min=0.0, max=1.0)
        return cdf

    @torch.no_grad()
    def compress(self, inputs):
        '''
        :param inputs: the y_hat tensor, shape [B, C, W, H]
        :return: a byte stream of y_hat and a side_info tuple
        '''
        (B, C, H, W) = inputs.shape
        assert B == 1, "Entropy coder only supports batch size one currently"
        # Get a series of symbols according to the minimum and maximum of y_hat
        symbol_max = torch.max(inputs).detach().to(torch.float)
        symbol_min = torch.min(inputs).detach().to(torch.float)
        symbol_samples = torch.arange(symbol_min, symbol_max + 1).to(inputs.device)
        symbol_samples = symbol_samples.reshape(1, 1, 1, -1).repeat(B, C, 1, 1)       # B, C, H, W
        # Get the pmf and cdf of the above symbols
        pmf = self.entropy_model.likelihood(symbol_samples).detach()
        pmf = torch.clamp(pmf, min=0.0, max=1.0)
        cdf = self.pmf_to_cdf(pmf)
        cdf = cdf.reshape(B, C, 1, 1, -1).repeat(1, 1, H, W, 1).to(torch.device('cpu'))
        # Shift the y_hat, make its elements start from 0
        inputs_norm = (inputs - symbol_min).to(torch.int16).to(torch.device('cpu'))
        # torchac only supports device('cpu')
        stream = torchac.encode_float_cdf(cdf, inputs_norm, needs_normalization=True)
        # The range of the symbols and the latent y shape needs to be transmitted
        side_info = (int(symbol_min), int(symbol_max), H, W)
        return stream, side_info

    @torch.no_grad()
    def decompress(self, stream, side_info, device=torch.device('cpu')):
        '''
        :param stream: the byte stream of coded y_hat; side_info: the side info tuple;
        device: device of self.bit_estimator
        :return: decoded y_hat as in torch.float32, shape
        '''
        (symbol_min, symbol_max, H, W) = side_info
        B, C = 1, self.entropy_model.num_channel
        # Get a series of symbols according to the minimum and maximum of y_hat
        symbol_samples = torch.arange(symbol_min, symbol_max + 1).to(device)
        symbol_samples = symbol_samples.reshape(1, 1, 1, -1).repeat(B, C, 1, 1)  # B, C, H, W
        # Get the pmf and cdf of the above symbols
        pmf = self.entropy_model.likelihood(symbol_samples).detach()
        pmf = torch.clamp(pmf, min=0.0, max=1.0)
        cdf = self.pmf_to_cdf(pmf)
        cdf = cdf.reshape(1, C, 1, 1, -1).repeat(1, 1, H, W, 1).to(torch.device('cpu'))
        # Get the decoded y_hat, which starts from 0
        y_hat_dec = torchac.decode_float_cdf(cdf, stream, needs_normalization=True).to(device).to(torch.float)
        # Shift to the right data range
        y_hat_dec += symbol_min
        return y_hat_dec

    def encode(self, inputs, filepath=''):
        '''
        :param inputs: the y_hat tensor, shape [B, C, W, H]; filepath: the output bitstream path, not write out in default
        :return: total bits to encode y_hat
        '''
        stream, side_info = self.compress(inputs)
        symbol_min, symbol_max, H, W = side_info
        for i in (symbol_min, symbol_max, H, W):
            stream += struct.pack('l', i)
        if filepath:
            with open(filepath, 'wb') as f:
                f.write(stream)
        return 8*len(stream)
        # print("Total bits: {:d}".format(8*len(stream)))

    def decode(self, filepath, device=torch.device('cpu')):
        '''
        :param filepath: teh path of btistream; device: device of self.bit_estimator
        :return: decoded y_hat, '' when fail
        '''
        assert os.path.exists(filepath), "Bitstream {} can ot be located".format(filepath)
        with open(filepath, 'rb') as f:
            stream = f.read()
        symbol_min = struct.unpack('l', stream[-16:-12])[0]
        symbol_max = struct.unpack('l', stream[-12:-8])[0]
        H = struct.unpack('l', stream[-8:-4])[0]
        W = struct.unpack('l', stream[-4:])[0]
        return self.decompress(stream[0:-16], (symbol_min, symbol_max, H, W), device=device)


class EntropyCoderGaussianMixture(EntropyCoder):
    def __init__(self, entropy_model):
        super(EntropyCoderGaussianMixture, self).__init__(entropy_model)

    @torch.no_grad()
    def compress(self, inputs, means, scales):
        assert (inputs.shape == means.shape) and (
                    inputs.shape == scales.shape), "Shape dismatch between y and gaussian means scales"
        (B, C, H, W) = inputs.shape
        assert B == 1, "Entropy coder only supports batch size one currently"
        # Get a series of symbols according to the minimum and maximum of y_hat
        symbol_max = torch.max(inputs).detach().to(torch.float)
        symbol_min = torch.min(inputs).detach().to(torch.float)

        # Get the pmf and cdf of the above symbols
        symbol_samples = torch.arange(symbol_min, symbol_max + 1).to(inputs.device)
        symbol_samples = symbol_samples.reshape((1, 1, 1, -1)).repeat((B, H * W, C, 1))     # B, H*W, C, L
        L = symbol_samples.shape[-1]

        means = means.reshape((B, C, H * W, 1)).permute((0, 2, 1, 3))
        scales = scales.reshape((B, C, H * W, 1)).permute((0, 2, 1, 3))

        raw_cdf = self.entropy_model(symbol_samples + 0.5, means, scales).detach()
        raw_cdf = raw_cdf + torch.linspace(1e-6, 1e-6 * L, steps=L).view(1, 1, 1, -1).to(raw_cdf.device)
        cdf_min = raw_cdf[:, :, :, 0].unsqueeze(-1)
        cdf_max = raw_cdf[:, :, :, L - 1].unsqueeze(-1)
        length = cdf_max - cdf_min
        cdf2 = (raw_cdf - cdf_min) / length
        cdf2 = torch.cat((torch.zeros(*cdf2.shape[0:3], 1).to(cdf2.device), cdf2), dim=-1).clamp(min=0.0, max=1.0)
        cdf2 = cdf2.reshape(B, H, W, C, -1).to(torch.device('cpu'))

        #######################
        # pmf = self.entropy_model.likelihood(symbol_samples, means, scales).detach()
        # pmf = torch.clamp(pmf, min=0.0, max=1.0)
        # cdf = self.pmf_to_cdf(pmf)
        # cdf = cdf.reshape(B, H, W, C, -1).to(torch.device('cpu'))
        #######################

        # Get the decoded y_hat, which starts from 0
        inputs = inputs.permute((0, 2, 3, 1))
        inputs_norm = (inputs - symbol_min).to(torch.int16).to(torch.device('cpu'))
        stream = torchac.encode_float_cdf(cdf2, inputs_norm, needs_normalization=True)
        # The range of the symbols and the latent y shape needs to be transmitted
        side_info = (int(symbol_min), int(symbol_max), H, W)
        return stream, side_info

    @torch.no_grad()
    def decompress(self, stream, side_info, means, scales, device=torch.device('cpu')):
        (symbol_min, symbol_max, H, W) = side_info
        assert (H == means.shape[2]) and (W == means.shape[3]) and (H == scales.shape[2]) and (
                    W == scales.shape[3]), "Shape dismatch between y and gaussian mean scales"
        B, C = 1, scales.shape[1]
        # Get a series of symbols according to the minimum and maximum of y_hat
        symbol_samples = torch.arange(symbol_min, symbol_max + 1).to(device)
        symbol_samples = symbol_samples.reshape((1, 1, 1, -1)).repeat((B, H * W, C, 1))  # B, H*W, C, L
        L = symbol_samples.shape[-1]

        # Permute the channel and H * W, since the dimension H * W is sparse
        means = means.reshape((B, C, H * W, 1)).permute((0, 2, 1, 3))
        scales = scales.reshape((B, C, H * W, 1)).permute((0, 2, 1, 3))

        # Get the cdf of the above symbols
        raw_cdf = self.entropy_model(symbol_samples + 0.5, means, scales).detach()
        raw_cdf = raw_cdf + torch.linspace(1e-6, 1e-6 * L, steps=L).view(1, 1, 1, -1).to(raw_cdf.device)
        cdf_min = raw_cdf[:, :, :, 0].unsqueeze(-1)
        cdf_max = raw_cdf[:, :, :, L - 1].unsqueeze(-1)
        length = cdf_max - cdf_min
        cdf2 = (raw_cdf - cdf_min) / length
        cdf2 = torch.cat((torch.zeros(*cdf2.shape[0:3], 1).to(cdf2.device), cdf2), dim=-1).clamp(min=0.0, max=1.0)
        cdf2 = cdf2.reshape(B, H, W, C, -1).to(torch.device('cpu'))

        ########################
        # pmf = self.entropy_model.likelihood(symbol_samples, means, scales).detach()
        # pmf = torch.clamp(pmf, min=0.0, max=1.0)
        # cdf = self.pmf_to_cdf(pmf)
        # cdf = cdf.reshape(B, H, W, C, -1).to(torch.device('cpu'))
        ########################

        y_hat_dec = torchac.decode_float_cdf(cdf2, stream, needs_normalization=True).to(device).to(torch.float)
        # Shift to the right data range
        y_hat_dec += symbol_min
        # Permute back: B, H, W, C -> B, C, H, W
        return y_hat_dec.permute((0, 3, 1, 2))


if __name__ == '__main__':
    bit_estimator = FactorizedModel((4, 192, 16, 16), K=4)
    entropy_coder = EntropyCoder(bit_estimator)
    y_hat = (torch.randn(4, 192, 16, 16) * 10).int()
    print(y_hat.type())
    entropy_coder.compress(y_hat)
