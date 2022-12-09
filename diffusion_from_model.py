import torch

from src import *

from src.BitDiffusion import *
import torchvision
import torchvision.transforms as transforms

device = torch.device("cpu")


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())
#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

file="models/diffusion_no_collapse_k=0.7.pt"
loaded=torch.load(file,map_location=device)
#torch.save(loaded.state_dict(), 'asd.pt')
loaded.self_conditioning=True
param=loaded.param
model = Unet(
    dim = param["Unet_dim"],
    channels = param["channels"],
    dim_mults = (1, 2, 4, 8),
    bits = BITS,
).to(device)


max_cross=param["max_cross"]
def cross_entropy(prediction, target, max_cross=max_cross):
    ce= target*torch.log2(prediction + 1e-8) + (1-target)*torch.log2(1-prediction + 1e-8)
    ce=torch.tanh(ce/max_cross)*max_cross
    return -torch.mean(ce)

def loss_function(predictions, target, max_cross=max_cross):
    target=target+.5
    predictions=predictions+.5
    probabilities=torch.sin(predictions*np.pi/2)**2
    return cross_entropy(probabilities,target, max_cross)

diffusion = BitDiffusion(model, image_size = 32, collapsing=param["collapsing"], schedule_function=cosine_schedule).to(device)
#diffusion.schedule=cosine_schedule
diffusion.losses = []
diffusion.timesteps=100
diffusion.self_conditioning=True

diffusion.load_state_dict(torch.load("asd.pt"))

batch_size=param['batch_size']
k=.7

images = decimal_to_qubits(testset[10][0].unsqueeze(0),bits=BITS)
t=diffusion.timesteps//3

alpha=diffusion.schedule(t)*torch.ones(len(images))

noised_images = gaussian_noise(images, alpha, k)

#sample=diffusion.sample(noised_images.shape, k)
sample=diffusion.denoise(noised_images, k, t, diffusion.timesteps)
sample=qubit_to_decimal(sample)
imshow(sample)
print('a')
