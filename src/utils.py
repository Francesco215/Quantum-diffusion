import torch
import torch.nn.functional as F


from einops import rearrange, reduce

import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


BITS = 8

# convert to qubit representations and back
def decimal_to_binary(img,bits=BITS):
    """
        expects image tensor with each x representing the polar angle of each qbit
    """
    assert img.shape[1]==3, f'the image should have 3 channels, the input has {img.shape[1]} channels'
    assert (img>=0).all().item() and (img<=1).all().item(), f'the image should be in the range [0,1), the input has values in the range [{img.min().item()},{img.max().item()}]'

    device = img.device

    x = (img * 255).int().clamp(0, 255)

    # powers of 2 [128,  64,  32,  16,   8,   4,   2,   1]
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1') #mask.shape == (8, 1, 1)
    x = rearrange(x, 'b c h w -> b c 1 h w') #batch, channel, height, width

    x = (x & mask) != 0
    x = rearrange(x, 'b c d h w -> b (c d) h w')
    return x

def binary_to_qubits(x):
    return x.float() - 0.5

def decimal_to_qubits(x,bits=BITS):
    """
        expects image tensor with each x representing the polar angle of each qbit
    """
    
    return decimal_to_binary(x,bits).float()-1/2

def qubit_to_binary(x):
    return (qubit_collapse(x) + 1/2).bool()

def binary_to_decimal(x, bits = BITS):
    device=x.device
    x=x.int()
    
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

def qubit_to_decimal(x, bits = BITS):
    """ expects x to be a polar angle in randians, returns a decimal number in the range [0, 1] """
    device = x.device

    x = (qubit_collapse(x)+1/2).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

def theta_to_prob(theta):
    return torch.sin((theta+1/2)*np.pi/2)**2

def qubit_collapse(x):
    return torch.bernoulli(theta_to_prob(x)) - 1/2

def cross_entropy(prediction,target):
    return -torch.mean(target*torch.log(prediction + 1e-8) + (1-target)*torch.log(1-prediction + 1e-8))


def probability_quantum_gaussian(mu:torch.Tensor, sigma:torch.Tensor, k=1) -> torch.Tensor:
    """Given a gaussian distribution of probability of the angle of the qubit with mean mu and standard deviation sigma,
        returns the probability of the qubit being in the state |1>

        Note:
        I choose to give the probability of the state |1> because the function torch.bernoulli()
        asks the probability of the state |1>
    Args:
        mu (torch.Tensor): mean of the gaussian distribution
        sigma (torch.Tensor): gaussian standard deviation
        k (float): it is a parameter that changes the way the gaussian noise id added. Defaults to 1. 

    Returns:
        torch.Tensor: the probability of being in the state |1>
    """
    sin=torch.sin(k*mu*np.pi)
    return (1-torch.exp(-np.pi**2*sigma**2*k**2/2)*sin)/2

def probablity_flip_gaussian(alpha:torch.Tensor,k=1) -> torch.Tensor:
    """TODO scrivere documentazione

    Args:
        alpha (torch.Tensor): _description_
        k (float): it is a parameter that changes the way the gaussian noise id added. Defaults to 1. 

    Returns:
        torch.Tensor: spin flip probability
    """

    mu=-torch.sqrt(alpha)/(2*k)
    sigma=torch.sqrt(torch.ones_like(alpha)-alpha)

    return 1-probability_quantum_gaussian(mu,sigma,k)

def bmult(batch_wise_vector, tensor) -> torch.Tensor:
    """Multiplies a vector for a tensor over the first dimention.
       it is used for multiplying each batch for a different number
    Args:
        batch_wise_vector (torch.Tensor or float): vector or scalar
        tensor (torch.Tensor): tensor

    Returns:
        torch.Tensor: tensor with the same dimentions of x
    """
    if type(batch_wise_vector)==float or type(batch_wise_vector)==int or len(batch_wise_vector)==1:
        return batch_wise_vector*tensor

    return torch.einsum("b,b...->b...",batch_wise_vector,tensor)

# old utils
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# functions to show an image
def imshow(img):
    plt.imshow(make_grid(img[:4]).cpu().detach().permute(1,2,0))
    plt.show()


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# utils for bit diffusion class
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))