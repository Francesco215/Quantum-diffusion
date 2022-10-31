import math

import torch
from torch.special import expm1
import torch.nn.functional as F


from einops import rearrange, reduce

import numpy as np

BITS = 8

# convert to qubit representations and back
def decimal_to_binary(img,bits=BITS):
    """
        expects image tensor with each x representing the polar angle of each qbit
    """
    assert img.shape[1]==3, f'the image should have 3 channels, the input has {img.shape[1]} channels'
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

#TODO: check if there is no floating point error
def qubit_to_binary(x):
    return (qubit_collapse(x) + 1/2).bool()

def qubit_to_decimal(x, bits = BITS):
    """ expects x to be a polar angle in randians, returns a decimal number in the range [0, 1] """
    device = x.device

    x = (qubit_collapse(x)+1/2).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = 8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

def theta_to_prob(theta, eps=1e-8):
    return torch.clamp(torch.sin((theta+1/2)*np.pi/2)**2, eps, 1 - eps)

def qubit_collapse(x):
    return torch.bernoulli(theta_to_prob(x)) - 1/2

def cross_entropy(prediction,target):
    return -torch.mean(target*torch.log(prediction + 1e-8) + (1-target)*torch.log(1-prediction + 1e-8))


def probability_quantum(sigma):
    """Returns the probability of having a spin flip as a function of the standard deviation of the gaussian

    Args:
        sigma (float): gaussian standard deviation

    Returns:
        float: the probability of having spin-flip
    """
    sin=np.sin(np.sqrt(1-sigma**2)*np.pi/2)
    return (1+np.exp(-2*sigma**2)*sin)/2

# old utils
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


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