import sys

sys.path.insert(0, './metrics/')
from factor_vae_metric import FactorVAEMetric
from mig import MIG
from betavae_metric import BetaVAEMetric
import numpy as np
import torch
import time
import logging
log = logging.getLogger(__name__)


class Evaluator(object):
	def __init__(self, data, config):
		self.data = data
		self.config = config
		self.device = torch.device('cuda:' + str(config.device_id))
		self.metric_eval = {'beta_vae': [], 'factor_vae': [], 'mig': []}

	def evaluate_model(self, model, epoch):
		start_time = time.time()
		beta_vae = BetaVAEMetric(self.data, self.device)
		factor_vae = FactorVAEMetric(self.data, self.device)
		mig = MIG(self.data, self.device)
		beta_vae_metric = beta_vae.compute_beta_vae(model, np.random.RandomState(self.config.random_seeds[0]), 64,
													2000,
													2000)
		factor_vae_metric = factor_vae.compute_factor_vae(model, np.random.RandomState(self.config.random_seeds[0]),
														  64,
														  2000, 2000, 2000)
		mutual_info_gap = mig.compute_mig(model, 2000, np.random.RandomState(self.config.random_seeds[0]),
										  batch_size=128)
		metrics = {'beta_vae': beta_vae_metric, 'factor_vae': factor_vae_metric, 'mig': mutual_info_gap[
			"discrete_mig"]}
		self.metric_eval['beta_vae'].append(metrics['beta_vae']["eval_accuracy"])
		self.metric_eval['factor_vae'].append(metrics['factor_vae']["eval_accuracy"])
		self.metric_eval['mig'].append(metrics['mig'])
		logging.info("Epochs  %d / %d Time taken %d sec B-VAE: %.3f, F-VAE %.3F, MIG : %.3f" % (epoch, self.config.epochs,
																						 time.time() - start_time,
																						 metrics['beta_vae'][
																							 "eval_accuracy"],
																						 metrics['factor_vae'][
																							 "eval_accuracy"],
																						 metrics['mig']))
		return self.metric_eval
