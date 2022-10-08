import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

import numpy as np

BITS = 8

from .utils import *

# convert to bit representations and back

def decimal_to_qubits(x,bits=BITS):
    """
        makes the wavefunction collapse
        expects image tensor with each x representing the polar angle of each qbit
    """
    device = x.device

    x = (x * 255).int().clamp(0, 255)

    # powers of 2 [128,  64,  32,  16,   8,   4,   2,   1]
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1') #mask.shape == (8, 1, 1)
    x = rearrange(x, 'b c h w -> b c 1 h w') #batch, channel, height, width

    bits = ((x & mask) != 0).float()*np.pi/2
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    return bits

def qubit_to_decimal(x, bits = BITS):
    """ expects x to be a polar angle in randians, returns a decimal number in the range [0, 1] """
    device = x.device

    x = torch.bernoulli(torch.sin(x)**2).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

def decimal_to_bits(x, bits = BITS): #ma a me sembra bits_to_decimal quello che fa
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    x = (x * 255).int().clamp(0, 255)

    # powers of 2 [128,  64,  32,  16,   8,   4,   2,   1]
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1') #mask.shape == (8, 1, 1)
    x = rearrange(x, 'b c h w -> b c 1 h w') #batch, channel, height, width

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits * 2 - 1
    return bits

def bits_to_decimal(x, bits = BITS):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)


# bit diffusion class

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def gamma_t(t, s: float = 0.008):
    return torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2 # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class BitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        use_ddim = False,
        noise_schedule = 'cosine',
        time_difference = 0.,
        bit_scale = 1.
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.bit_scale = bit_scale

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

    @property
    def device(self):
        return next(self.model.parameters()).device


    @torch.no_grad()
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        times= torch.linspace(1., 0., self.timesteps + 1, device = device)

        img = torch.randn(shape, device = device)

        x_start = None

        for i in range in tqdm(len(times)-1, desc = 'sampling loop time step'):

            # scheduling
            gamma_now = gamma_t(times[i])
            gamma_next = gamma_t(times[i+1])


            x_start = self.model(img, gamma_now, x_start)

            # get predicted noise



        return bits_to_decimal(img)

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times

        #times = torch.zeros((batch,), device = device).float().uniform_(0, 0.999)
        times=torch.rand([batch],device=device)

        # convert image to bit representation

        img = decimal_to_qubits(img)# * self.bit_scale

        # noise sample

        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = sigma * noise + img# * alpha

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, noise_level).detach_()

        # predict and take gradient step

        pred = self.model(noised_img, noise_level, self_cond)

        return pred
        #return F.mse_loss(pred, img)
