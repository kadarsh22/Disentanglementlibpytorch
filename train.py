import numpy as np
import torch
import torch.nn as nn
import time
import os
import random
import logging

log = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class Trainer(object):

	def __init__(self, dsprites, config):
		super(Trainer, self).__init__()
		self.data = dsprites
		self.config = config
		self.device = torch.device('cuda:' + str(config['device_id']))
		self.train_loader = self._get_training_data()
		self.train_hist_vae = {'loss': [], 'bce_loss': [], 'kld_loss': []}
		self.train_hist_gan = {'d_loss': [], 'g_loss': [], 'info_loss': []}

	def train_vae(self, model, optimizer, epoch):
		start_time = time.time()
		bce_loss, kld_loss, total_loss = 0, 0, 0
		for images in self.train_loader:
			images = images.to(self.device)
			optimizer.zero_grad()
			loss, out = model(images)
			loss[0].backward()
			optimizer.step()
			bce_loss = bce_loss + loss[2].item()
			kld_loss = kld_loss + loss[1].item()
			total_loss = total_loss + loss[0].item()
		logging.info("Epochs  %d / %d Time taken %d sec Loss : %.3f BCELoss: %.3f, KLDLoss %.3F" % (
			epoch, self.config['epochs'], time.time() - start_time, total_loss / len(self.train_loader),
			bce_loss / len(self.train_loader), kld_loss / len(self.train_loader)))
		self.train_hist_vae['loss'].append(total_loss / self.config['batch_size'])
		self.train_hist_vae['bce_loss'].append(bce_loss / self.config['batch_size'])
		self.train_hist_vae['kld_loss'].append(kld_loss / self.config['batch_size'])
		return model, self.train_hist_vae, (optimizer,)

	def _get_training_data(self):
		images = torch.from_numpy(self.data.images)
		train_loader = torch.utils.data.DataLoader(images, batch_size=self.config['batch_size'], shuffle=True)
		return train_loader

	@staticmethod
	def set_seed(seed):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)

	def train_gan(self, model, d_optimizer, g_optimizer, info_optimizer, epoch):
		start_time = time.time()
		d_loss, g_loss, info_loss = 0, 0, 0
		BCE_loss = nn.BCELoss().to(self.device)
		MSE_loss = nn.MSELoss().to(self.device)
		y_real_, y_fake_ = torch.ones(self.config['batch_size'], 1).to(self.device), torch.zeros(
			self.config['batch_size'], 1).to(self.device)
		model.encoder.to(self.device)
		model.decoder.to(self.device)
		y_cont = torch.FloatTensor(self.config['batch_size'], self.config['latent_dim']).uniform_(-1,
																								  1).to(self.device)
		for iter, images in enumerate(self.train_loader):
			images = images.type(torch.FloatTensor).to(self.device)
			z_ = torch.randn((self.config['batch_size'], self.config['noise_dim']),device=self.device)
			c_= torch.rand(self.config['batch_size'], self.config['latent_dim'],device=self.device) * 2 - 1
			input_vec = torch.cat((z_, c_), dim=1)

			d_optimizer.zero_grad()

			_, D_real = model.encoder(images)
			D_real_loss = BCE_loss(D_real, y_real_)

			G_ = model.decoder(input_vec)
			_, D_fake, = model.encoder(G_)
			D_fake_loss = BCE_loss(D_fake, y_fake_)
			D_loss = D_real_loss + D_fake_loss

			D_loss.backward(retain_graph=True)
			d_optimizer.step()

			g_optimizer.zero_grad()

			G_ = model.decoder(input_vec)
			D_cont, D_fake = model.encoder(G_)
			G_loss = BCE_loss(D_fake, y_real_)

			G_loss.backward(retain_graph=True)
			g_optimizer.step()

			cont_loss = MSE_loss(D_cont, y_cont)
			cont_loss.backward()
			info_optimizer.step()

			d_loss = d_loss + D_loss.item()
			g_loss = g_loss + G_loss.item()
			info_loss = info_loss + cont_loss.item()
		logging.info("Epochs  %d / %d Time taken %d sec Info_Loss : %.3f D_Loss: %.3f, G_Loss %.3F" % (
			epoch, self.config['epochs'], time.time() - start_time, info_loss / len(self.train_loader),
			g_loss / len(self.train_loader), d_loss / len(self.train_loader)))

		self.train_hist_gan['d_loss'].append(d_loss / len(self.train_loader))
		self.train_hist_gan['g_loss'].append(g_loss / len(self.train_loader))
		self.train_hist_gan['info_loss'].append(info_loss / len(self.train_loader))

		return model, self.train_hist_gan, (d_optimizer, g_optimizer, info_optimizer)
