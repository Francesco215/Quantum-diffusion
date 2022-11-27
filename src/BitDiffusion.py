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
        schedule: Callable=linear_schedule,
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
    def sample(self, shape, k,  timesteps=None):
        """Generates an image from pure noise

        Args:
            shape (torch.Tensor): the shape of the images to generate. (b,c,h,w)
            device (torch.device, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: the generated images
        """
        assert len(shape) == 4, f'the images should have the four dimentions b,c,h,w, instead it has {len(shape)}'
        assert shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto

        timesteps=default(timesteps,self.timesteps)
        return generate_from_noise(self.model,self.reverse_step, shape, timesteps, self.schedule, self.device, k, self.collapsing)

    #WIP
    @torch.no_grad()
    def denoise(self, images, k, time=None, timesteps=None):
        assert len(images.shape) == 4, f'the images should have the four dimentions b,c,h,w, instead it has {len(images.shape)}'
        assert images.shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto

        timesteps=default(timesteps,self.timesteps)
        time = default(time, timesteps)
        return denoise_images(self.model, self.reverse_step, images, time, timesteps, self.schedule, k, self.collapsing)


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

        _, _, h, w, img_size, = *img.shape, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        cond_img = None
        if self_conditioning and random() < 0.5:
            with torch.no_grad(): #testare dopo con il gradiente
                cond_img = self.model(img, noise_level).detach_()
                if self.collapsing: cond_img=qubit_collapse(cond_img)

        # predict
        pred = self.model(img, noise_level, cond_img)
        
        return pred

