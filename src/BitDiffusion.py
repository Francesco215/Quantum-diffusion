from random import random

import torch
from torch import nn

from tqdm.auto import tqdm

import numpy as np

BITS = 8

from .utils import *



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

            x_start=ddim_step(img, times[i], times[i + 1])  #TODO: add self conditioning

        return qubit_to_decimal(x_start)

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

















































#Utils for diffusion



def ddim_step(x_t, t_now, t_next, model, conditioning=None):
    """
        A single step of diffusion denoising implicit model
        args:
            x_t: the target
            x_pred: the prediction
            t_now: the current time step
            t_next: the next time step

        returns:
            the next prediction
    """

    # scheduling
    gamma_now = gamma_t(t_now)
    gamma_next = gamma_t(t_next)

    #prediction of the target
    x_pred = model(x_t, gamma_now, conditioning)

    #error
    eps= (x_t-torch.sqrt(gamma_now)*x_pred)/torch.sqrt(1-gamma_now)

    #update
    x_next=x_t-torch.sqrt(1-gamma_next)*eps

    return x_next

def ddpm_step(x_t, t_now, t_next, model, conditioning=None):
    """
        A single step of diffusion denoising probabilistic model
        args:
            x_t: the target
            x_pred: the prediction
            t_now: the current time step
            t_next: the next time step

        returns:
            the next prediction
    """

    # scheduling
    gamma_now = gamma_t(t_now)
    gamma_next = gamma_t(t_next)
    alpha=gamma_now/gamma_next
    sigma=torch.sqrt(1-alpha)


    z=torch.normal(0,1,size=x_t.shape,device=x_t.device)

    #prediction of the target
    x_pred = model(x_t, gamma_now, conditioning)

    #error
    eps = (x_t - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(1 - gamma_now)

    #update
    x_next = x_t - ((1-alpha)/torch.sqrt(alpha*(1 - gamma_now))) * eps + sigma * z

    return x_next