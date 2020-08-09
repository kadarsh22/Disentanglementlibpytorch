import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
	def __init__(self, latent_dim=5):
		super(VAE, self).__init__()

		self.latent_dim = latent_dim
		self.nc = 1

		self.cnn1_en = nn.Conv2d(self.nc, 32, 4, 2, 1)
		self.cnn2_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.cnn3_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.cnn4_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.linear1_en = nn.Linear(32 * 4 * 4, 256)
		self.linear2_en = nn.Linear(256, 256)
		self.z_mean = nn.Linear(256, 2 * self.latent_dim)
		self.act = nn.ReLU(inplace=True)

		self.linear1_dec = nn.Linear(self.latent_dim, 256)
		self.linear2_dec = nn.Linear(256, 256)
		self.linear3_dec = nn.Linear(256, 32 * 4 * 4)
		self.cnn1_dec = nn.ConvTranspose2d(32, 32, 4, 2, 1)
		self.cnn2_dec = nn.ConvTranspose2d(32, 32, 4, 2, 1)
		self.cnn3_dec = nn.ConvTranspose2d(32, 32, 4, 2, 1)
		self.cnn4_dec = nn.ConvTranspose2d(32, self.nc, 4, 2, 1)

		self.tanh = nn.Tanh()

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
		return z_parameters[:, :self.latent_dim], z_parameters[:, self.latent_dim:]

	def reparametrize(self, mu, logvar):
		std = logvar.div(2).exp()
		eps = Variable(std.data.new(std.size()).normal_())
		return mu + std * eps

	def decoder(self, z):
		out = self.act(self.linear1_dec(z))
		out = self.act(self.linear2_dec(out))
		out = self.act(self.linear3_dec(out)).view((-1, 32, 4, 4))
		out = self.act(self.cnn1_dec(out))
		out = self.act(self.cnn2_dec(out))
		out = self.act(self.cnn3_dec(out))
		out = self.cnn4_dec(out)
		return out.view(-1, 64, 64)

	def lossfun(self, x_in, x_out, z_mu, z_logvar):
		beta = 8
		x_in = x_in.type(torch.cuda.FloatTensor)
		x_out = x_out.view(-1, 64, 64)
		bce_loss = F.binary_cross_entropy_with_logits(x_out, x_in, size_average=False)
		kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
		loss = (bce_loss + beta * kld_loss) / x_out.size(0)
		return loss, beta * kld_loss / x_out.size(0), bce_loss / x_out.size(0)

	def forward(self, x):
		z_mean, z_stddev = self.encoder(x)
		z = self.reparametrize(z_mean, z_stddev)
		out = self.decoder(z)
		loss = self.lossfun(x, out, z_mean, z_stddev)

		return loss, z_mean
