from .utils import probablity_flip_gaussian, bmult, qubit_collapse, BITS
import numpy as np

import torch




#this part is to add noise
def bernoulli_noise(img:torch.Tensor, alpha:torch.Tensor, k:float=1) -> torch.Tensor:
    """takes a batch of images and adds to each one of them a bernoulli noise.
        The Bernulli noise can be applied only if the img is in the bit-wise representation
    Args:
        img (torch.Tensor): images to add noise to
        alpha (torch.Tensor): the sqrt(1-alpha) is the variance of the gaussian noise
        k (float): it is a parameter that changes the way the gaussian noise id added. Defaults to 1. 

    Returns:
        torch.Tensor: The noised images
    """
    assert alpha.shape[0]==img.shape[0], f'alpha must have the same size as the batch size of img, alpha has {alpha.shape[0]} and img has {img.shape[0]}'
    assert img.dtype==torch.bool, f'img must be in the bit-wise representation, img has dtype {img.dtype}'

    p_flip=probablity_flip_gaussian(alpha, k)

    bernulli_prob = bmult(p_flip, torch.ones_like(img))
    noise = torch.bernoulli(bernulli_prob).bool()

    return img ^ noise

def gaussian_noise(img:torch.Tensor, alpha:torch.Tensor, k:float=1):
    """Takes a batch of images and adds to each one of them a gaussian.
        The gaussian noise can be applied only if the img is in the qubit representation
    Args:
        img (torch.Tensor): images to add noise to
        alpha (torch.Tensor): the sqrt(1-alpha) is the variance of the gaussian noise
        k (float): it is a parameter that changes the way the gaussian noise id added. Defaults to 1. 

    Returns:
        torch.Tensor: The noised images
    """
    assert alpha.shape[0]==img.shape[0], f'alpha must have the same size as the batch size of img, alpha has {alpha.shape[0]} and img has {img.shape[0]}'
    assert img.dtype == torch.float, f'img must be in the qubit representation, img has dtype {img.dtype}'

    mu, sigma = torch.sqrt(alpha), torch.sqrt(1-alpha)*k

    noise = torch.randn_like(img).to(img.device)
    while torch.any(torch.isnan(noise)):
        noise = torch.randn_like(img).to(img.device)

    #       x*sqrt(alpha) + noise*sqrt(1-alpha)
    return bmult(mu, img) + bmult(sigma, noise)


#This part is for the scheduling of the alphas
#TODO: check if this function is consistend with the reverse step definition
def linear_schedule(t:float ,t_max:float, bits=BITS):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/bits

    return (t/t_max+s)/(1+s)

#TODO: check if this function is consistend with the reverse step definition
def cosine_schedule(t:float ,t_max:float, bits=BITS):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/bits

    #TODO: check if using the torch implementation for the cosine function is better
    return np.cos((t/t_max+s)/(1+s)*np.pi/2)**2









#this part is for the denoising
def denoise_images(forward, reverse_step_function, x, time, timesteps, schedule, k, collapsing, self_conditioning) -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        forward (Callable): the forward function of the diffusion model diffusion.forward()
        x (torch.Tensor): the images to denoise (b,c,h,w) 
            the first dimention is the batch
            the second dimention represents the channels, it must be equal to 3*BITS
            the third and fourth dimention represent the height and width of the image
        time (float): the current timestep
        timesteps (float): the number of total timesteps to count
        schedule (Callable): the function to use for the scheduling of the alphas
        k (float): it is a parameter that changes the way the gaussian noise id added
        collapsing (bool): if True then the qubits are collapsed after each timestep
    Returns:
        torch.Tensor: The generated images
    """
    #TODO: check the last step
    alpha_old = schedule(time, timesteps)
    alpha_next=schedule(time-1,timesteps)*torch.ones(len(x)).to(x.device)
    for t in range(time-2,0,-1):
        noise_level=probablity_flip_gaussian(alpha_next, k)

        # predict
        x_0 = forward(x, noise_level, self_conditioning)


        alpha_old=alpha_next
        alpha_next=schedule(t,timesteps)*torch.ones(len(x)).to(x.device)

        # reverse step
        x=reverse_step_function(x, x_0, alpha_old, alpha_next)
        
        if collapsing: x=qubit_collapse(x)
    x=forward(x,noise_level)
    if collapsing: x=qubit_collapse(x)
    return x


# Utils for diffusion
def generate_from_noise(forward, reverse_step_function, shape, timesteps, schedule, device, k, collapsing, self_conditioning) -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        forward (Callable): the forward function of the diffusion model diffusion.forward()
        reverse_step_function (Callable): the function to use for the reverse step
        shape (float): the shape of the tensor to generate. (b,c,h,w)
            the first dimention is the batch
            the second dimention represents the channels, it must be equal to 3*BITS
            the third and fourth dimention represent the height and width of the image
        timesteps (float): the number of total timesteps to count
        schedule (Callable): the function to use for the scheduling of the alphas
        device (torch.Device): the device to use
        k (float): it is a parameter that changes the way the gaussian noise id added
        collapsing (bool): if True then the qubits are collapsed after each timestep
    Returns:
        torch.Tensor: The generated images
    """

    if collapsing: x=torch.poisson(0.5*torch.ones(shape)).to(device).float() - 0.5
    else: x=gaussian_noise(torch.zeros(shape), 1, k)
    return denoise_images(forward, reverse_step_function, x, timesteps, timesteps, schedule, k, collapsing, self_conditioning)



#this part defines the reverse step
def reverse_step(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float, sigma:float) -> torch.Tensor:
    """Does the reverse step, you must calculate the prediction of the original image x_0 separatelly. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        x_0 (torch.tensor): prediction of the original image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper
        sigma (int or torch.Tensor, optional): Noise of the reverse step,
            if sigma = 0 then it is a DDIM step,
            if sigma = sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.

    Returns:
        torch.Tensor: reverse step
    """
    mean=bmult(torch.sqrt(alpha_next),x_0)
    
    dx=x-bmult(torch.sqrt(alpha_old),x_0)
    const=torch.sqrt((1-sigma**2-alpha_next)/(1-alpha_old))
    
    mean+=bmult(const,dx)

    #      mean + normal(mean=0,std=sigma, size=x.shape)
    return mean + bmult(sigma, torch.normal(0, 1, size=x.shape, device=x.device))
    

def reverse_step_epsilon(x: torch.tensor, epsilon:torch.tensor, alpha_old:float, alpha_next:float, sigma:float) -> torch.Tensor:
    """Does the reverse step, you must calculate the noise epsilon separately. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        epsilon (torch.tensor): the prediction of the noise added to x
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper
        sigma (int or torch.Tensor, optional): Noise of the reverse step,
            if sigma = 0 then it is a DDIM step,
            if sigma = sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.

    Returns:
        torch.Tensor: reverse step
    """
    #dx = -sqrt(1-alpha_old)*epsilon
    dx = bmult(torch.sqrt( 1-alpha_old ), - epsilon)
    
    # mean = (x+dx)*sqrt(alpha_old/alpha_next) + epsilon*sqrt(1 - sigma**2 - alpha_old)
    mean = bmult(torch.sqrt( alpha_old/alpha_next ),x + dx)
    mean += bmult(torch.sqrt( 1 - sigma**2 - alpha_old ) , epsilon)
    #      mean + normal(mean=0,std=sigma, size=x.shape)
    return mean + bmult(sigma, torch.normal(0, 1, size=x.shape, device=x.device))



def reverse_DDIM(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float) -> torch.Tensor:
    """Calculates the reverse step. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        x_0 (torch.tensor): prediction of the original image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper

    Returns:
        torch.Tensor: reverse step
    """
    return reverse_step(x,x_0,alpha_old,alpha_next,0)

def reverse_DDPM(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float) -> torch.Tensor:
    """Calculates the reverse step. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        x_0 (torch.tensor): prediction of the original image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper

    Returns:
        torch.Tensor: reverse step
    """
    sigma=torch.sqrt((1-alpha_old)/alpha_old)
    return reverse_step(x,x_0,alpha_old,alpha_next,sigma)