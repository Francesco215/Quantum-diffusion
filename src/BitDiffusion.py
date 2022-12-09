from .utils import default, qubit_collapse, BITS
from .diffusion_utils import denoise_images, generate_from_noise, linear_schedule, reverse_DDIM
from random import random

import torch
from torch import nn


from typing import Callable


class BitDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,  # ma a che serve sto asterisco?
        image_size,
        schedule_function: Callable=linear_schedule,
        timesteps: int=1000,
        reverse_step: Callable=reverse_DDIM,
        collapsing: bool=True,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size
        self.schedule_function = schedule_function
        self.timesteps = timesteps
        self.collapsing = collapsing
        self.reverse_step = reverse_step

    def schedule(self, t: int, timesteps=None) -> float:
        timesteps = default(timesteps, self.timesteps)
        return self.schedule_function(t, timesteps)

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
    def denoise(self, images: torch.Tensor,
                k:float,
                time:float=None,
                timesteps:float=None,
                self_conditioning:bool=None) -> torch.Tensor:
        """Denoises the images 

        Args:
            images (torch.Tensor): images to denoise   
            k (float): 
            time (float, optional): time passed in the noising process
            timesteps (float, optional): total number of timesteps. Defaults to self.timesteps

        Returns:
            torch.Tensor: Denoised images
        """
        assert len(images.shape) == 4, f'the images should have the four dimentions b,c,h,w, instead it has {len(images.shape)}'
        assert images.shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto

        timesteps=default(timesteps,self.timesteps)
        time = default(time, timesteps)
        self_conditioning = default(self_conditioning, self.self_conditioning)
        return denoise_images(self.forward, self.reverse_step, images, time, timesteps, self.schedule, k, self.collapsing,self_conditioning)


    def forward(self, img:torch.Tensor,
                noise_level:torch.Tensor,
                self_conditioning:bool = False
                ) -> torch.Tensor:
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
        pred = self.model(cond_img, noise_level, img)
        
        return pred

