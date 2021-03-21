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
        beta_vae = BetaVAEMetric(self.data.metric_data, self.device, self.config)
        factor_vae = FactorVAEMetric(self.data.metric_data, self.device, self.config)
        mig = MIG(self.data.metric_data, self.device,self.config)
        beta_vae_metric = beta_vae.evaluate(model)
        factor_vae_metric = factor_vae.evaluate(model)
        mutual_info_gap = mig.evaluate(model)
        metrics = {}
        for regressor in ["LassoCV","RandomForestCV"]:
            DCI_metric = DCIMetric(self.data.metric_data, regressor=regressor)
            metrics["DCI_{}".format(regressor)] = DCI_metric.evaluate(model)
        metrics.update({'beta_vae': beta_vae_metric, 'factor_vae': factor_vae_metric, 'mig': mutual_info_gap})
        self.metric_eval['beta_vae'].append(metrics['beta_vae'])
        self.metric_eval['factor_vae'].append(metrics['factor_vae'])
        self.metric_eval['mig'].append(metrics['mig']["MIG_metric"])
        logging.info('Mig Vector')
        logging.info(metrics['mig']['mig_factor_wise'])
        logging.info(metrics['DCI_RandomForestCV']['DCI_RandomForestCV_disent_metric_detail'])
        logging.info('Disentanglement Vector')
        logging.info(metrics['DCI_RandomForestCV']['DCI_RandomForestCV_disent_metric_detail'])
        logging.info('completeness_vector')
        logging.info(metrics['DCI_RandomForestCV']['DCI_RandomForestCV_complete_metric_detail'])
        logging.info(
            "Epochs  %d / %d Time taken %d sec B-VAE: %.3f, F-VAE %.3F, MIG : %.3f Disentanglement: %.3f "
            "Completeness: "
            "%.3f  " % (
                epoch, self.config['epochs'],
                time.time() - start_time,
                metrics['beta_vae'],
                metrics['factor_vae'][0],
                metrics['mig']["MIG_metric"], metrics['DCI_RandomForestCV']['DCI_RandomForestCV_disent_metric'],
                metrics['DCI_RandomForestCV']['DCI_RandomForestCV_complete_metric']

            ))
        return self.metric_eval
