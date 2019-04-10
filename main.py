import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.nn import init
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import numpy as np
from tensorboardX import SummaryWriter
from torchsummary import summary

import models
import utils
import argparse

desc = "Pytorch implementation of VAE'"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dim_z', type=int, help='Dimensionality of latent variables', default=20)
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate of ADAM optimizer')
parser.add_argument('--epochs', type=int, default=110, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--gpu', type=bool, default=False, help='Use GPU?')
parser.add_argument('--type', type=str, default='VAE', help='VAE, VAE_cnn')
parser.add_argument('--n_gpu', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=100)

args = parser.parse_args()

""" GPU """

if_cuda = args.gpu and torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.n_gpu)

# Enable CUDA, set tensor type and device
if if_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

""" Directory loading """

Model_dir = 'Models/'

if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)

Data_dir = 'Data/'

if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)

Summary_dir = 'Summary'

if not os.path.exists(Summary_dir):
    os.makedirs(Summary_dir)

Fig_dir = 'Figs/'

if not os.path.exists(Fig_dir):
    os.makedirs(Fig_dir)

""" Summary """
writer = SummaryWriter(Summary_dir)

""" Data loader """

train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.gpu)

val_loader = DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.gpu)

X_val, _ = next(iter(val_loader))

""" Model"""

if args.type == "VAE":
    model = models.VAE(args.dim_z, 512, 256).type(dtype)
else:
    model = models.VAE_cnn(args.dim_z).type(dtype)

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def loss_vae(input, output, mu, log_var):
    liklihood = F.binary_cross_entropy(output, input.view(-1, 784), reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return liklihood + kl, liklihood, kl


def train_test(model, train_loader, val_loader, loss_function, epoch):
    model.train()
    l_train = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        out, mu, log_var = model(x)
        loss, lk, kld = loss_function(x, out, mu, log_var)
        l_train += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tLk: {:.6f} \tKLD: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(x), lk.item() / len(x),
                       kld.item() / len(x)))
            writer.add_scalar('train/loss', loss.item() / len(x))

    model.eval()
    l_val = 0
    for batch_idx, (x, _) in enumerate(val_loader):
        x = x.to(device)
        out, mu, log_var = model(x)
        loss_val, _, _ = loss_function(x, out, mu, log_var)
        l_val += loss_val.item()

    return l_train / len(train_loader.dataset), l_val / len(val_loader.dataset)


best_loss = np.inf

for epoch in range(1, args.epochs + 1):
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    l_train, l_val = train_test(model, train_loader, val_loader, loss_vae, epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, l_train))

    if l_val < best_loss:
        best_loss = l_val
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        },
            Model_dir + 'VAE_{}_z_{}_epch_{}_lr_{}.pt'.format(args.type, args.dim_z, epoch, args.lr))

        writer.add_scalar('test/loss', l_val, epoch)

    print('==== Testing. Loss: {:.4f} ====\n'.format(l_val))


def reconstruct(model, device, dtype, loader_val):
    model.eval()
    X_val, _= next(iter(loader_val))
    X_val = X_val.type(dtype)
    X_val = X_val.to(device)
    X_hat_val, _ , _ =model(X_val)

    X_val = X_val[:10].cpu().view(10 * 28, 28)
    X_hat_val = X_hat_val[:10].cpu().view(10 * 28, 28)
    comparison = torch.cat((X_val, X_hat_val), 1).view(10 * 28, 2 * 28)
    return comparison


comparison = reconstruct(model, device, dtype, val_loader)
save_image(comparison,
           Fig_dir + 'VAE_comparison_{}_z_{}_epch_{}_lr_{}.png'.format(args.type, args.dim_z, args.epochs, args.lr))





