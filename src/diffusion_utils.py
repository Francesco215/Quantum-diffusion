from .utils import *
import torch




#this part is to add noise
def bernoulli_noise(img, alpha) -> torch.Tensor:
    """takes a batch of images and adds to each one of them a bernoulli noise.
        The Bernulli noise can be applied only if the img is in the bit-wise representation
    Args:
        img (torch.Tensor): images to add noise to
        alpha (torch.Tensor): the sqrt(alpha) is the variance of the gaussian noise

    Returns:
        torch.Tensor: The noised images
    """
    assert alpha.shape[0]==img.shape[0], f'alpha must have the same size as the batch size of img, alpha has {alpha.shape[0]} and img has {img.shape[0]}'
    assert img.dtype==torch.bool, f'img must be in the bit-wise representation, img has dtype {img.dtype}'

    sigma=torch.sqrt(torch.ones_like(alpha)-alpha)
    p_flip=probability_quantum(sigma)

    bernulli_prob = torch.einsum("b, bchw -> bchw", p_flip, torch.ones_like(img))
    noise = torch.bernoulli(bernulli_prob).bool()

    return img ^ noise

def gaussian_noise(img, alpha):
    """takes a batch of images and adds to each one of them a bernoulli noise.
        The gaussian noise can be applied only if the img is in the qubit representation
    Args:
        img (torch.Tensor): images to add noise to
        alpha (torch.Tensor): the sqrt(1-alpha) is the variance of the gaussian noise

    Returns:
        torch.Tensor: The noised images
    """
    assert alpha.shape[0]==img.shape[0], f'alpha must have the same size as the batch size of img, alpha has {alpha.shape[0]} and img has {img.shape[0]}'
    assert img.dtype ==torch.float, f'img must be in the qubit representation, img has dtype {img.dtype}'

    mu, s = torch.sqrt(alpha), torch.sqrt(1-alpha)
    noise = torch.randn_like(img)
    #       x*sqrt(alpha)                           +           noise*sqrt(1-alpha)
    return torch.einsum("b, bchw -> bchw", mu, img) + torch.einsum("b, bchw -> bchw", s, noise)


#This part is for the scheduling of the alphas
#TODO: check if this function is consistend with the reverse step definition
def cosine_schedule(t:float ,t_max:float ):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/BITS

    #TODO: check if using the torch implementation for the cosine function is better
    return np.cos((t/t_max+s)/(1+s)*np.pi/2)**2


#this part is for the denoising
def denoise_images(model, img, time, timesteps, sigma=0,schedule=None) -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        model (nn.Module): the model to use for generation
        img (torch.Tensor): the images to denoise (b,c,h,w) 
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
    assert img.shape[1] == 3*BITS, f'channels must be {3*BITS}' #TODO: controllare che sia corretto

    schedule=default(schedule,cosine_schedule)

    alpha_next=schedule(time,timesteps)
    x=img.clone()#.to(img.device)
    for t in range(time,-1,-1):
        epsilon=model(x,t)
        alpha_old=alpha_next
        alpha_next=schedule(t,timesteps)
        x=reverse_step(x,epsilon,alpha_old,alpha_next,sigma)

    return x
    
# Utils for diffusion
def generate_from_noise(model, shape, timesteps, sigma=0, schedule=None, device='cpu') -> torch.Tensor:
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

    x=torch.poisson(0.5*torch.ones(shape),device=device)
    return denoise_images(model,x,timesteps,timesteps,sigma,schedule)



#this part defines the reverse step
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