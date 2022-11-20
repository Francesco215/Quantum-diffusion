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
        collapsing: bool=True,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size
        self.schedule = schedule
        self.timesteps = timesteps
        self.collapsing = collapsing
        self.reverse_step = reverse_step

    @property #is this useful?
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample(self, shape, timesteps=None):
        """Generates an image from pure noise

        Args:
            shape (torch.Tensor): the shape of the images to generate. (b,c,h,w)
            device (torch.device, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: the generated images
        """
        timesteps=default(timesteps,self.timesteps)
        return generate_from_noise(self.model,self.reverse_step, shape, timesteps, self.schedule, self.device)



    def forward(self, img:torch.Tensor,
                noise_level:torch.Tensor,
                self_conditioning:bool = False,
                *args, **kwargs) -> torch.Tensor:
        """This is an estiamate of the target image

        Args:
            img (torch.Tensor): the image to do the prediction of the target image
            noise (torch.Tensor): noise applied to each image. It can be calculated with probability_flip_gaussian()
            self_conditioning (bool, optional): wether or not to applu the self-conditioning to the algorithm. Defaults to False.

        Returns:
            torch.Tensor: The predition of the target image
        """

        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        cond_img = None
        if self_conditioning and random() < 0.5:
            with torch.no_grad(): #testare dopo con il gradiente
                cond_img = self.model(img, noise_level).detach_()

        # predict
        pred = self.model(img, noise_level, cond_img)
        
        return pred

