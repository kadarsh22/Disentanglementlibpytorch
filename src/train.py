import time
import os
import random
import logging
from utils import *

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

	def train_gan(self, model, optimizer, epoch):
		d_optimizer = optimizer[0]
		g_optimizer = optimizer[1]


	@staticmethod
	def set_seed(seed):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)

	def _get_training_data(self):
		images = self.data.images
		train_loader = torch.utils.data.DataLoader(images, batch_size=self.config['batch_size'], shuffle=True)
		return train_loader
