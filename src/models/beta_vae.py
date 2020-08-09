import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
	def __init__(self, latent_dim=10):
		super(VAE, self).__init__()

		self.fc1_en = nn.Linear(4096, 1200)
		self.fc2_en = nn.Linear(1200, 1200)
		self.fc3_mean = nn.Linear(1200, latent_dim)
		self.fc3_stddev = nn.Linear(1200, latent_dim)
		self.act = nn.ReLU(inplace=True)

		self.decoder = nn.Sequential(nn.Linear(latent_dim, 1200), nn.Tanh(),
									 nn.Linear(1200, 1200), nn.Tanh(),
									 nn.Linear(1200, 1200), nn.Tanh(),
									 nn.Linear(1200, 4096))

		self.latent_dim = latent_dim

	def encoder(self, x):
		x = x.type(torch.cuda.FloatTensor)
		out = x.view(-1, 4096)

		out = self.fc1_en(out)
		out = self.act(out)

		out = self.fc2_en(out)
		out = self.act(out)

		z_mean = self.fc3_mean(out)
		z_stddev = self.fc3_stddev(out)

		return z_mean, z_stddev

	def reparametrize(self, mu, logvar):
		std = logvar.div(2).exp()
		eps = Variable(std.data.new(std.size()).normal_())
		return mu + std * eps

	def decode(self, z):
		out = self.decoder(z)
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
		out = self.decode(z)
		loss, kld_loss, bce_loss = self.lossfun(x, out, z_mean, z_stddev)

		return (loss, kld_loss, bce_loss), (out, z_mean, z_stddev)
