import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.nn import Parameter
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
        # self.linear4 = nn.Linear(in_features=128, out_features=128)
        # self.linear5 = nn.Linear(in_features=128, out_features=self.dim_c_cont)
        self.disc_layer = nn.Linear(in_features=128, out_features=1)




    def forward_no_spectral(self, z):
        z = z.type(torch.cuda.FloatTensor)
        out = F.leaky_relu(self.conv1(z.view(-1, 1, 64, 64)),negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
        out = F.leaky_relu(self.linear1(out.view(-1, 64 * 4 * 4)), negative_slope=0.2)
        probability = F.sigmoid(self.disc_layer(out))
        probability = probability.squeeze()
        c_vec = F.leaky_relu(self.linear2(out),negative_slope=0.2)
        c_cont = self.linear3(c_vec)
        # similarity_vec = F.leaky_relu(self.linear4(out),negative_slope=0.2)
        # similarity = self.linear5(similarity_vec)
        return c_cont , probability

    def forward(self, z):
        z = z.type(torch.cuda.FloatTensor)

        normalised_weights_conv1 = self.spectral_normed_weight(self.conv1)
        setattr(self.conv1, 'weight', torch.nn.Parameter(normalised_weights_conv1))
        out = F.leaky_relu(self.conv1(z.view(-1, 1, 64, 64)),negative_slope=0.2)

        normalised_weights_conv2 = self.spectral_normed_weight(self.conv2)
        setattr(self.conv2, 'weight', torch.nn.Parameter(normalised_weights_conv2))
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)

        normalised_weights_conv3 = self.spectral_normed_weight(self.conv3)
        setattr(self.conv3, 'weight',torch.nn.Parameter(normalised_weights_conv3))
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)

        normalised_weights_conv4 = self.spectral_normed_weight(self.conv4)
        setattr(self.conv4, 'weight',torch.nn.Parameter(normalised_weights_conv4))
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)

        normalised_linear1_weight = self.spectral_normed_weight(self.linear1)
        setattr(self.linear1, 'weight', torch.nn.Parameter(normalised_linear1_weight))
        out = F.leaky_relu(self.linear1(out.view(-1, 64 * 4 * 4)), negative_slope=0.2)

        probability = F.sigmoid(self.disc_layer(out))
        probability = probability.squeeze()

        normalised_linear2_weight = self.spectral_normed_weight(self.linear2)
        setattr(self.linear2, 'weight', torch.nn.Parameter(normalised_linear2_weight))
        c_vec = F.leaky_relu(self.linear2(out),negative_slope=0.2)

        normalised_linear3_weight = self.spectral_normed_weight(self.linear3)
        setattr(self.linear3, 'weight', torch.nn.Parameter(normalised_linear3_weight))
        c_cont = self.linear3(c_vec)
        # normalised_linear4_weight = spectral_normed_weight(self.linear4.weight)
        # self.linear4.weight.data = normalised_linear4_weight
        # similarity_vec = F.leaky_relu(self.linear4(out),negative_slope=0.2)
        # normalised_linear5_weight = spectral_normed_weight(self.linear5.weight)
        # self.linear5.weight.data = normalised_linear5_weight
        # similarity = self.linear5(similarity_vec)
        return c_cont ,probability

    def l2normalize(self,v, eps=1e-12):
        return v / (v.norm() + eps)

    def spectral_normed_weight(self ,module_inp):
        try:
            u = getattr(module_inp, 'weight' + "_u")
            v = getattr(module_inp, 'weight' + "_v")
            w = getattr(module_inp, 'weight' + "_bar")
        except AttributeError:
            w = getattr(module_inp, 'weight')

            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]

            u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            u.data = l2normalize(u.data)
            v.data = l2normalize(v.data)
            w_bar = Parameter(w.data)

            module_inp.register_parameter('weight' + "_u", u)
            module_inp.register_parameter('weight' + "_v", v)
            module_inp.register_parameter('weight' + "_bar", w_bar)


        height = w.data.shape[0]
        for _ in range(1):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

class InfoGan(object):
    def __init__(self, config):
        super(InfoGan, self).__init__()

        self.decoder = Generator(dim_z=config['noise_dim'], dim_c_cont=config['latent_dim'])
        self.encoder = Discriminator(dim_c_cont=config['latent_dim'])

    def dummy(self):
        print('This is a dummy function')

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)