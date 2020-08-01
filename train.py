import numpy as np
import torch
import time
import os
import random
import logging
log = logging.getLogger(__name__)


class Trainer(object):

	def __init__(self, dsprites, config):
		super(Trainer, self).__init__()
		self.data = dsprites
		self.config = config
		self.device = torch.device('cuda:' + str(config.device_id))
		self.train_loader = self._get_training_data()
		self.train_hist_vae = {'loss': [], 'bce_loss': [], 'kld_loss': []}

	def train_vae(self, model, optimizer, epoch):
		start_time = time.time()
		bce_loss, kld_loss, total_loss = [], [], []
		for images in self.train_loader:
			images = images.to(self.device)
			optimizer.zero_grad()
			loss, out = model(images)
			loss[0].backward()
			optimizer.step()
			bce_loss.append(loss[2].item())
			kld_loss.append(loss[1].item())
			total_loss.append(loss[0].item())
		logging.info("Epochs  %d / %d Time taken %d sec Loss : %.3f BCELoss: %.3f, KLDLoss %.3F" % (
			epoch, self.config.epochs, time.time() - start_time, sum(total_loss) / len(total_loss),
			sum(bce_loss) / len(bce_loss), sum(kld_loss) / len(kld_loss)))
		self.train_hist_vae['loss'].append(sum(total_loss) / len(total_loss))
		self.train_hist_vae['bce_loss'].append(sum(bce_loss) / len(bce_loss))
		self.train_hist_vae['kld_loss'].append(sum(kld_loss) / len(kld_loss))
		return model, self.train_hist_vae

	def _get_training_data(self):
		images = torch.from_numpy(self.data.images)
		train_loader = torch.utils.data.DataLoader(images, batch_size=self.config.batch_size, shuffle=True)
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

	def train_gan(self):
		raise NotImplementedError
