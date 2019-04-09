

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class VAE(nn.Module):


    def __init__(self,dim_z,d_hidden1, d_hidden2):
        super(VAE, self).__init__()
        self.train_step = 0
        self.best_loss = np.inf
        self.fc_enc1 = nn.Linear(784, d_hidden1)
        self.fc_enc2 = nn.Linear(d_hidden1, d_hidden2)
        self.fc_enc3_1 = nn.Linear(d_hidden2, dim_z)
        self.fc_enc3_2 = nn.Linear(d_hidden2, dim_z)


        self.fc_dec1 = nn.Linear(dim_z, d_hidden2)
        self.fc_dec2 = nn.Linear(d_hidden2, d_hidden1)
        self.fc_dec3 = nn.Linear(d_hidden1, 784)

    def encode(self, x):
        h1 = F.relu(self.fc_enc1(x))
        h2 = F.relu(self.fc_enc2(h1))
        h3_1 = self.fc_enc3_1(h2)
        h3_2 = self.fc_enc3_1(h2)
        return h3_1, h3_2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc_dec1(z))
        h5 = F.relu(self.fc_dec2(h4))
        output = torch.sigmoid(self.fc_dec3(h5))
        return output

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE_cnn(nn.Module):

    def __init__(self,dim_z):
        super(VAE_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=dim_z)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=dim_z)
        self.relu = nn.ReLU()

        # For decoder

        # For mu
        self.fc1 = nn.Linear(in_features=dim_z, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu = F.elu(self.fc11(x))
        mu = self.fc12(mu)

        logvar = F.elu(self.fc21(x))
        logvar = self.fc22(logvar)

        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = F.sigmoid(self.conv_t2(x))

        return x.view(-1, 784)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar



