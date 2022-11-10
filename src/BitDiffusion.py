from .utils import *
from .diffusion_utils import *
from random import random

import torch
from torch import nn

from tqdm.auto import tqdm

from typing import Callable

BITS = 8

class BitDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,  # ma a che serve sto asterisco?
        image_size,
        schedule: Callable=cosine_schedule,
        timesteps: int=1000,
        reverse_step: Callable=reverse_DDIM,
        time_difference=0.,
        collapsing: bool=True,
        noise_fn: Callable=gaussian_noise,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.schedule = schedule
        self.timesteps = timesteps
        self.collapsing = collapsing
        self.noise_fn = noise_fn
        self.reverse_step = reverse_step

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400
        self.time_difference = time_difference

    @property #is this useful?
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample(self, shape):
        """Generates an image from pure noise

        Args:
            shape (torch.Tensor): the shape of the images to generate. (b,c,h,w)
            device (torch.device, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: the generated images
        """

        return generate_from_noise(self.model,self.reverse_step, shape, self.timesteps, self.schedule, self.device)



    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # convert image to bit representation
        img = decimal_to_qubits(img)

        # noise sample
        alpha = torch.rand(batch, device=device)
        #t = self.gamma_t(t)

        noised_img = self.noise_fn(img, alpha)

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, alpha).detach_()

        # predict
        pred = self.model(noised_img, alpha, self_cond)
        
        return pred, noised_img

