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
		images = self.data.images
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

	def train_gan(self, model, optimizer, epoch):
		d_optimizer = optimizer[0]
		g_optimizer = optimizer[1]
		start_time = time.time()
		d_loss, g_loss, info_loss = 0, 0, 0
		model.encoder.to(self.device)
		model.decoder.to(self.device)

		adversarial_loss = torch.nn.BCELoss()
		continuous_loss = torch.nn.MSELoss()

		for iter, images in enumerate(self.train_loader):

			images = images.type(torch.FloatTensor).to(self.device)


			z = torch.randn(self.config['batch_size'], self.config['noise_dim'], device=self.device)
			c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'],
								device=self.device) * 2 - 1
			z = torch.cat((z, c_cond), dim=1)

			d_optimizer.zero_grad()

			prob_real = model.encoder(images)[1]
			label_real = torch.full(	(self.config['batch_size'],), 1,dtype=torch.float32, device=self.device)
			loss_D_real = adversarial_loss(prob_real, label_real)

			loss_D_real.backward()

			data_fake = model.decoder(z)
			prob_fake_D = model.encoder(data_fake.detach())[1]

			label_fake = torch.full((self.config['batch_size'],), 0,dtype=torch.float32, device=self.device)
			loss_D_fake = adversarial_loss(prob_fake_D, label_fake)

			loss_D_fake.backward()
			loss_D = loss_D_real + loss_D_fake

			d_optimizer.step()

			g_optimizer.zero_grad()

			latent_code , prob_fake = model.encoder(data_fake)

			loss_G = adversarial_loss(prob_fake, label_real)
			loss_c_cont = continuous_loss(c_cond, latent_code)

			loss_info = loss_G + loss_c_cont
			loss_info.backward()

			g_optimizer.step()

			d_loss = d_loss + loss_D.item()
			g_loss = g_loss + loss_G.item()
			info_loss = info_loss + loss_c_cont.item()
		#
		logging.info("Epochs  %d / %d Time taken %d sec Info_Loss : %.3f D_Loss: %.3f, G_Loss %.3F" % (
			epoch, self.config['epochs'], time.time() - start_time, info_loss / len(self.train_loader),
			g_loss / len(self.train_loader), d_loss / len(self.train_loader)))

		self.train_hist_gan['d_loss'].append(d_loss / len(self.train_loader))
		self.train_hist_gan['g_loss'].append(g_loss / len(self.train_loader))
		self.train_hist_gan['info_loss'].append(info_loss / len(self.train_loader))

		return model, self.train_hist_gan, (d_optimizer, g_optimizer)
