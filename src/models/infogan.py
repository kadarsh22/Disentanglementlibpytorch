import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

'''
Generator Model Definition
'''

class Generator(nn.Module):
    def __init__(self, dim_z, n_c_disc, dim_c_disc, dim_c_cont):
        super(Generator, self).__init__()
        self.dim_latent = dim_z + n_c_disc * dim_c_disc + dim_c_cont
        self.fc1 = nn.Linear(in_features=self.dim_latent,
                             out_features=1024,
                             bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,
                             out_features=7*7*128,
                             bias=False)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.upconv3 = nn.ConvTranspose2d(in_channels=128,
                                          out_channels=64,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)

    def forward(self, z):
        # Layer 1: [-1, dim_latent] -> [-1, 1024]
        z = F.relu(self.bn1(self.fc1(z)))

        # Layer 2: [-1, 1024] -> [-1, 7*7*128]
        z = F.relu(self.bn2(self.fc2(z)))

        # Shape Change: [-1, 7*7*128] -> [-1, 128, 7, 7]
        z = z.view(-1, 128, 7, 7)

        # Layer 3: [-1, 128, 7, 7] -> [-1, 64, 14, 14]
        z = F.relu(self.bn3(self.upconv3(z)))

        # Layer 4: [-1, 64, 14, 14] -> [-1, 1, 28, 28]
        img = torch.sigmoid(self.upconv4(z))

        return img



class Discriminator(nn.Module):
    '''Shared Part of Discriminator and Recognition Model'''

    def __init__(self, n_c_disc , dim_c_disc, dim_c_cont):
        super(Discriminator, self).__init__()
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        self.n_c_disc = n_c_disc
        # Shared layers
        self.module_shared = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 128*7*7),
            nn.Linear(in_features=128*7*7, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Layer for Disciminating
        self.module_D = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

        self.module_Q = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.latent_disc = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=10),
            Reshape(-1, 1, 10),
            nn.Softmax(dim=2)
        )

        self.latent_cont_mu = nn.Linear(
            in_features=128, out_features=5)

        self.latent_cont_var = nn.Linear(
            in_features=128, out_features=5)

    def forward(self, z):
        out = self.module_shared(z)
        probability = self.module_D(out)
        probability = probability.squeeze()
        internal_Q = self.module_Q(out)
        c_cont_mu = self.latent_cont_mu(internal_Q)
        c_cont_var = torch.exp(self.latent_cont_var(internal_Q))
        c_disc_logits = self.latent_disc(internal_Q)
        return probability, c_disc_logits, c_cont_mu, c_cont_var

class CRDiscriminator(nn.Module):
    def __init__(self, z_dim):
        super(CRDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z).squeeze()


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class InfoGan(object):
    def __init__(self, config):
        super(InfoGan, self).__init__()

        self.decoder = Generator(dim_z=config['noise_dim'],n_c_disc=1,dim_c_disc=config['discrete_dim'], dim_c_cont=config['latent_dim'])
        self.encoder = Discriminator(n_c_disc=1,dim_c_disc = config['discrete_dim'],dim_c_cont=config['latent_dim'])
        self.cr_disc = CRDiscriminator(z_dim=5)

    def dummy(self):
        print('This is a dummy function')
