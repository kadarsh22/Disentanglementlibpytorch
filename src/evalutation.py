import sys

sys.path.insert(0, './metrics/')
from factor_vae_metric import FactorVAEMetric
from mig import MIG
from betavae_metric import BetaVAEMetric
from dci_metric import DCIMetric
import numpy as np
import torch
import time
import logging

log = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.device = torch.device('cuda:' + str(config['device_id']))
        self.metric_eval = {'beta_vae': [], 'factor_vae': [], 'mig': []}
        self.mode = config['full_data']

    def evaluate_model(self, model, epoch):
        if self.config['model_arch'] == 'gan':
            model.encoder.to(self.device)
            model.decoder.to(self.device)
            model.encoder.eval()
            model.decoder.eval()
        start_time = time.time()
        beta_vae = BetaVAEMetric(self.data, self.device)
        factor_vae = FactorVAEMetric(self.data, self.device, self.config)
        DCI_metric = DCIMetric(self.data, self.device)
        dci = DCI_metric.compute_dci(model)
        mig = MIG(self.data, self.device)
        beta_vae_metric = beta_vae.compute_beta_vae(model, np.random.RandomState(self.config['random_seed']),
                                                    batch_size=64,
                                                    num_train=10000, num_eval=5000)
        factor_vae_metric = factor_vae.compute_factor_vae(model, np.random.RandomState(self.config['random_seed']),
                                                          batch_size=64, num_train=10000, num_eval=5000,
                                                          num_variance_estimate=10000)
        mutual_info_gap = mig.compute_mig(model, num_train=10000, batch_size=128)
        dci_average = (dci['disentanglement']+dci['completeness'] +dci['informativeness'])/3
        metrics = {'beta_vae': beta_vae_metric, 'factor_vae': factor_vae_metric, 'mig': mutual_info_gap,
                   'dci_metric': dci_average }
        self.metric_eval['beta_vae'].append(metrics['beta_vae']["eval_accuracy"])
        self.metric_eval['factor_vae'].append(metrics['factor_vae']["eval_accuracy"])
        self.metric_eval['mig'].append(metrics['mig'])
        logging.info('Disentanglement Vector')
        logging.info(dci['disentanglement_vector'])
        logging.info('completeness_vector')
        logging.info(dci['completeness_vector'])
        logging.info('informativeness_vector')
        logging.info(dci['informativeness_vector'])
        logging.info(
            "Epochs  %d / %d Time taken %d sec B-VAE: %.3f, F-VAE %.3F, MIG : %.3f Disentanglement: %.3f Completeness: "
            "%.3f Informativeness: %.3f " % (
                epoch, self.config['epochs'],
                time.time() - start_time,
                metrics['beta_vae'][
                    "eval_accuracy"],
                metrics['factor_vae'][
                    "eval_accuracy"],
                metrics['mig'], dci['disentanglement'],
                dci['completeness'], dci['informativeness']
            ))
        return self.metric_eval
