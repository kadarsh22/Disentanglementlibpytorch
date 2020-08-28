import torch
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self, output_dim=40):
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

	def forward(self, x):
		z_mean = self.encoder(x)
		return z_mean