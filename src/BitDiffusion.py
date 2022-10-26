from .utils import *
from random import random

import torch
from torch import nn

from tqdm.auto import tqdm

BITS = 8

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
        
        return pred, noised_img



def bernoulli_noise(img, alpha) -> torch.Tensor:
    """takes a batch of images and adds to each one of them a bernoulli noise

    Args:
        img (torch.Tensor): images to add noise to
        alpha (torch.Tensor): the sqrt(alpha) is the variance of the gaussian noise

    Returns:
        torch.Tensor: The noised images
    """
    assert alpha.shape[0]==img.shape[0], f'alpha and img must have the same batch size, alpha has {alpha.shape[0]} and img has {img.shape[0]}'

    p_flip=probability_quantum(torch.ones_like(alpha)-alpha)
    bernulli_prob = torch.einsum("b, bchw -> bchw", p_flip, torch.ones_like(img))
    noise = torch.bernoulli(bernulli_prob)

    return ((img.bool()) ^ (noise.bool())).float()


def gaussian_noise(img, t):
    mu, s = torch.sqrt(1-t), torch.sqrt(t)
    noise = torch.randn_like(img)

    return torch.einsum("b, bchw -> bchw", mu, img) + torch.einsum("b, bchw -> bchw", s, noise)


def denoise_images(model, image, time, timesteps, sigma=0, device="cpu") -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        model (nn.Module): the model to use for generation
        shape (float): the shape of the tensor to generate. (b,c,h,w)
            the first dimention is the batch
            the second dimention represents the channels, it must be equal to 3*BITS
            the third and fourth dimention represent the height and width of the image
        timesteps (float): the number of total timesteps to count
        sigma (int, optional): Noise of the reverse step,
            if sigma==0 then it is a DDIM step,
            if sigma==sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.
        device (optional): the device to use
    Returns:
        torch.Tensor: The generated images
    """
    assert image.shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto
    #WIP

    
# Utils for diffusion
def generate_from_noise(model, shape, timesteps, sigma=0, device='cpu') -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        model (nn.Module): the model to use for generation
        shape (float): the shape of the tensor to generate. (b,c,h,w)
            the first dimention is the batch
            the second dimention represents the channels, it must be equal to 3*BITS
            the third and fourth dimention represent the height and width of the image
        timesteps (float): the number of total timesteps to count
        sigma (int, optional): Noise of the reverse step,
            if sigma==0 then it is a DDIM step,
            if sigma==sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.
        device (optional): the device to use
    Returns:
        torch.Tensor: The generated images
    """
    assert shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto
    x=torch.poisson(0.5*torch.ones(shape),device=device)
    alpha_next=0

    for t in range(timesteps):
        alpha_old=alpha_next
        alpha_next=cosine_schedule(t,timesteps)
        epsilon=model(x,t)
        x=reverse_step(x,epsilon,alpha_old,alpha_next,sigma)

    return x



def cosine_schedule(t:float ,t_max:float ):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/BITS

    #TODO: check if using torch for the sqrt function is better
    return np.cos((t/t_max+s)/(1+s)*np.pi/2)**2


def reverse_step(x: torch.tensor, epsilon:torch.tensor, alpha_old:float, alpha_next:float ,sigma=0) -> torch.Tensor:
    """Calculates the reverse step. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        epsilon (torch.tensor): prediction of the final image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper
        sigma (int, optional): Noise of the reverse step,
            if sigma==0 then it is a DDIM step,
            if sigma==sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.

    Returns:
        torch.Tensor: reverse step
    """

    mean = ( x - torch.sqrt( 1 - alpha_old )*epsilon ) * torch.sqrt( alpha_old/alpha_next )
    mean += torch.sqrt( 1-alpha_old-sigma**2 ) * epsilon

    return mean + torch.normal(0, sigma, size=x.shape, device=x.device)


