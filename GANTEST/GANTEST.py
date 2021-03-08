import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

# standardization code
#standardizator = transforms.Compose([
#    transforms.ToTensor(), 
#    transforms.Normalize(mean=(0.5,0.5,0.5),  # 3 for RGB channels이나 실제론 gray scale
#                         std=(0.5,0.5,0.5))]) # 3 for RGB channels이나 실제론 gray scale

standardizator = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5),  # 3 for RGB channels이나 실제론 gray scale
                         std=(0.5))])

# MNIST dataset
train_data = dsets.MNIST(root='data/', train=True , transform=standardizator, download=True)
test_data  = dsets.MNIST(root='data/', train=False, transform=standardizator, download=True)


batch_size=200
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

import numpy as np
from matplotlib import pyplot as plt

def imshow(img):
    img = (img+1)/2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

example_mini_batch_img, example_mini_batch_label = next(iter(train_data_loader))
imshow_grid(example_mini_batch_img[0:16, :, :])

d_noise  = 100
d_hidden = 256

def sample_z(batch_size=1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)

G = nn.Sequential(
    nn.Linear(d_noise, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 28*28),
    nn.Tanh()
    ).to(device)

# 노이즈 생성하기
z = sample_z()
# 가짜 이미지 생성하기
img_fake = G(z).view(-1,28,28)
# 이미지 출력하기
imshow(img_fake.squeeze().cpu().detach())

#Batch Size만큼 노이즈 생성하여 그리드로 출력하기
z = sample_z(batch_size)
img_fake = G(z)
imshow_grid(img_fake)

D = nn.Sequential(
    nn.Linear(28*28, d_hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, d_hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden,1),
    nn.Sigmoid()).to(device)

print(G(z).shape)
print(D(G(z)).shape)
print(D(G(z)[0:5]).transpose(0,1))

criterion = nn.BCELoss()

def run_epoch(generator, disciminator, _optimizer_g, _optimizer_d):

    generator.train()
    discriminator.train()

    for img_batch, label_batch in train_data_loader:

        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        # ========================================================= #
        # maximize V(discriminator, generator) = optimize discriminator (setting k to be 1) #
        # ========================================================= #

        # init optimizer
        _optimizer_d.zero_grad()

        p_real = disciminator(img_batch.view(-1, 28*28))
        p_fake = disciminator(generator(sample_z(batch_size, d_nosie)))

        # ========================================================= #
        # Loss computation (soley based on the paper) #
        # ========================================================= #
        loss_real = -1 * torch.log(p_real) # -1 for gradient ascending
        loss_fake = -1 * torch.log(1.-p_fake) # -1 for gradient ascending
        loss_d = (loss_real + loss_fake).mean()

        # ========================================================= #
        # Loss computation (based on Cross Entropy) #
        # ========================================================= #
        # loss_d = criterion(p_real, torch.ones_like(p_real).to(device)) + \ #
        #          criterion(p_fake, torch.zeros_like(p_real).to(device))    #

        # Update parameters
        loss_d.backward()
        _optimizer_d.step()

        # ========================================================= #
        # minimize V(discriminator, generator)
        # ========================================================= #

        # init optimizer
        _optimizer_g.zero_grad()

        p_fake = discriminator(generator(sample_z(batch_size, d_noise)))

        # ========================================================= #
        # Loss computation (soley vased on the paper) #
        # ========================================================= #

        # insted of: torch.log(1.-p_fake, torch.ones_like(p_fake).to(device)) #

        loss_g.backward()

        # Update parameters
        _optimizer_g.step()