import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

'''
Generator Model Definition
'''


class Generator(nn.Module):
    def __init__(self, dim_z=5, dim_c_cont=5):
        super(Generator, self).__init__()
        self.dim_latent = dim_z + dim_c_cont
        self.fc1 = nn.Linear(in_features=self.dim_latent, out_features=128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4 * 4 * 64)
        self.bn2 = nn.BatchNorm1d(4 * 4 * 64)
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.upconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        # Layer 1: [-1, dim_latent] -> [-1, 128]
        z = self.bn1(F.relu(self.fc1(z)))

        # Layer 2: [-1, 1024] -> [-1, 4*4*64]
        z = self.bn2(F.relu(self.fc2(z)))

        # Shape Change: [-1, 4*4*64] -> [-1, 64, 4, 4]
        z = z.view(-1, 64, 4, 4)

        # Layer 3: [-1, 64, 4, 4] -> [-1, 64, 8, 8]
        z = self.bn3(F.leaky_relu(self.upconv3(z), negative_slope=0.2))

        # Layer 4: [-1, 64, 8, 8] -> [-1, 32, 16, 16]
        z = self.bn4(F.leaky_relu(self.upconv4(z), negative_slope=0.2))

        # Layer 5: [-1, 32, 16, 16] -> [-1, 32, 32,32]
        z = self.bn5(F.leaky_relu(self.upconv5(z), negative_slope=0.2))

        # Layer 6: [-1, 32, 32, 32] -> [-1, 1, 64, 64]
        img = torch.sigmoid(self.upconv6(z))

        return img


class Discriminator(nn.Module):

    def __init__(self, dim_c_cont=5):
        super(Discriminator, self).__init__()
        self.dim_c_cont = dim_c_cont
        # Shared layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=self.dim_c_cont)

        self.module_shared_spectral = nn.Sequential(
            spectral_norm(self.conv1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(self.conv2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(self.conv3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(self.conv4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Reshape(-1, 64 * 4 * 4),
            spectral_norm(self.linear1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.module_shared_no_spectral = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.conv2,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.conv3,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.conv4,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Reshape(-1, 64 * 4 * 4),
            self.linear1,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.module_Q_no_spectral = nn.Sequential(
            self.linear2,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.linear3)

        # Layer for Discriminating
        self.module_D = nn.Sequential(nn.Linear(in_features=128, out_features=1), nn.Sigmoid())

        self.module_Q_spectral = nn.Sequential(
            spectral_norm(self.linear2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(self.linear3)
        )

    def forward_no_spectral(self, z):
        z = z.type(torch.cuda.FloatTensor)
        out = self.module_shared_no_spectral(z.view(-1, 1, 64, 64))
        probability = self.module_D(out)
        probability = probability.squeeze()
        c_cont = self.module_Q_no_spectral(out)
        return c_cont, probability

    def forward(self, z):
        z = z.type(torch.cuda.FloatTensor)
        out = self.module_shared_spectral(z.view(-1, 1, 64, 64))
        probability = self.module_D(out)
        probability = probability.squeeze()
        c_cont = self.module_Q_spectral(out)
        return c_cont, probability


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Classifier(nn.Module):
    def __init__(self, output_dim=5):
        super(Classifier, self).__init__()

        self.latent_dim = output_dim
        self.nc = 1

        self.cnn1_en = nn.Conv2d(self.nc, 32, 4, 2, 1)
        self.cnn2_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.cnn3_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.cnn4_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.linear1_en = nn.Linear(32 * 4 * 4, 256)
        self.linear2_en = nn.Linear(256, 256)
        self.z_mean = nn.Linear(256, self.latent_dim)
        self.act = nn.ReLU(inplace=True)

    def encoder(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.view(-1, self.nc, 64, 64)
        out = self.act(self.cnn1_en(x))
        out = self.act(self.cnn2_en(out))
        out = self.act(self.cnn3_en(out))
        out = self.act(self.cnn4_en(out)).view(-1, 32 * 4 * 4)
        out = self.act(self.linear1_en(out))
        out = self.act(self.linear2_en(out))
        z_parameters = self.z_mean(out)
        return z_parameters

    def forward(self, positive,negative,query):
        pos = self.encoder(positive)
        neg = self.encoder(negative)
        que = self.encoder(query)
        return pos ,neg , que


class InfoGan(object):
    def __init__(self, config):
        super(InfoGan, self).__init__()

        self.decoder = Generator(dim_z=config['noise_dim'], dim_c_cont=config['latent_dim'])
        self.encoder = Discriminator(dim_c_cont=config['latent_dim'])
        self.oracle_shape = Classifier()
        self.oracle_size = Classifier()
        self.oracle_orient = Classifier()
        self.oracle_xpos = Classifier()
        self.oracle_ypos = Classifier()

    def dummy(self):
        print('This is a dummy function')
