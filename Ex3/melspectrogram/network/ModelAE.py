import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch import nn
from abc import abstractmethod


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(9, 16, 4, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 4 * 4 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
        unflattened_size=(32, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 16, 3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 9, 4, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#
#         # Encoder
#         self.conv1 = nn.Conv2d(9, 24, 3, padding=1)
#         self.conv2 = nn.Conv2d(24, 10, 3, padding=1)
#         self.conv2 = nn.Conv2d(24, 10, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         # Decoder
#         self.t_conv1 = nn.ConvTranspose2d(10, 24, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(24, 9, 2, stride=2)
#
#     def enc(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#
#         return x
#
#     def dec(self, x):
#         x = F.relu(self.t_conv1(x))
#         x = F.sigmoid(self.t_conv2(x))
#
#         return x
#
#     def forward(self, x):
#         x = self.enc(x)
#         x = self.dec(x)
#
#         return x