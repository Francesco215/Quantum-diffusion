from .utils import *
from random import random

import torch
from torch import nn
from torch.distributions.exponential import Exponential

from tqdm.auto import tqdm

BITS = 8


def bernoulli_noise(img, t):
    bernulli_prob = torch.einsum("b, bchw -> bchw", t*0.5, torch.ones_like(img))
    noise = torch.bernoulli(bernulli_prob)

    return ((img.bool()) ^ (noise.bool())).float()

def gaussian_noise(img, t):
    mu, s = torch.sqrt(1-t), torch.sqrt(t)
    noise = torch.randn_like(img)

    return mu*img + s*noise


class BitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,  # ma a che serve sto asterisco?
        image_size,
        gamma_t=gamma_t,
        timesteps=1000,
        d_step='ddpm',
        time_difference=0.,
        collapsing=True,
        noise_fn=bernoulli_noise,
        noise_before=False
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.gamma_t = gamma_t
        self.timesteps = timesteps
        self.collapsing = collapsing
        self.noise_fn = noise_fn
        self.noise_before = noise_before

       # choose the type of diffusion step
        if d_step == 'ddpm':
            self.d_step = ddpm_step
        elif d_step == 'ddim':
            self.d_step = ddim_step
        else:
            raise ValueError(
                f'd_step must be ddpm or ddim, you passed "{d_step}"')

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400
        self.time_difference = time_difference

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample(self, shape, conditioning=False, device=None):

        device = default(device, self.device)
        times = torch.linspace(0., 1., self.timesteps + 1, device=device)

        x = torch.rand(shape, device=device)
        for i in tqdm(range(len(times)-1), desc='sampling loop time step'):
            if self.collapsing:
                x = qubit_collapse(x)

            x = self.d_step(x, times[i:i+1], times[i+1:i+2], self.model,
                            conditioning=conditioning, gamma_t=self.gamma_t)
        return qubit_to_decimal(x)

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # noise sample
        t = torch.rand(batch, device=device)
        t = self.gamma_t(t)

        if self.noise_before:
            img = gaussian_noise(img, t)

        # convert image to bit representation
        img = decimal_to_qubits(img)

        noised_img = self.noise_fn(img, t)

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, t).detach_()

        # predict
        pred = self.model(noised_img, t, self_cond)

        return pred


# Utils for diffusion
def ddim_step(x, t_now, t_next, model, conditioning=None, gamma_t=gamma_t):
    """
        A single step of diffusion denoising probabilistic model
        args:
            x_t: the image
            t_now: the current time step
            t_next: the next time step
            conditioning: if not None, the conditioning tensor (aka. the previous prediction)
        returns:
            the next prediction
    """

    # scheduling
    gamma_now = gamma_t(t_now)
    gamma_next = gamma_t(t_next)

    # conditioning
    self_cond = None
    if conditioning:
        self_cond = model(x, gamma_now).detach_()

    # prediction of the target
    x_pred = model(x, gamma_now, self_cond)

    # error
    eps = (x-torch.sqrt(gamma_now)*x_pred)/torch.sqrt(1-gamma_now)

    # update
    x_next = x-torch.sqrt(1-gamma_next)*eps

    return x_next


def ddpm_step(x, t_now, t_next, model, conditioning=None, gamma_t=gamma_t):
    """
        A single step of diffusion denoising probabilistic model
        args:
            x_t: the image
            t_now: the current time step
            t_next: the next time step
            conditioning: if not None, the conditioning tensor (aka. the previous prediction)
        returns:
            the next prediction
    """

    # scheduling
    gamma_now = gamma_t(t_now)
    gamma_next = gamma_t(t_next)

    # conditioning
    self_cond = None
    if conditioning:
        self_cond = model(x, gamma_now).detach_()

    # prediction of the target
    x_pred = model(x, gamma_now, self_cond)
    alpha = gamma_now/gamma_next
    sigma = torch.sqrt(torch.abs(1-alpha))

    # error
    eps = (x - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(1 - gamma_now)
    z = torch.normal(0, 1, size=x.shape, device=x.device)

    # update
    m1 = ((1-alpha)/torch.sqrt(alpha*(1 - gamma_now)))
    x_next = x - m1 * eps + sigma * z

    return x_next
