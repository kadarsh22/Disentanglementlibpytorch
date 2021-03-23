import numpy as np
import logging
from sklearn.metrics import mutual_info_score
import torch

import sklearn


class MIG(object):
    """
            Implementation of the metric in: MIG
    """

    def __init__(self, metric_data, device_id,config):
        super(MIG, self).__init__()
        self.metric_data = metric_data
        self.device_id = device_id
        self.config = config

    def discretize(self, data, num_bins=20):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        discretized = np.zeros_like(data)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(
                data[:, i],
                np.histogram(data[:, i], num_bins)[1][:-1])
        return discretized

    def mutual_info(self, data1, data2):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        n1 = data1.shape[1]
        n2 = data2.shape[1]
        mi = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                mi[i, j] = mutual_info_score(
                    data2[:, j], data1[:, i])
        return mi

    def entropy(self, data):
        """ Adapted from:
            https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/utils.py
        """
        num_factors = data.shape[1]
        entr = np.zeros(num_factors)
        for i in range(num_factors):
            entr[i] = mutual_info_score(data[:, i], data[:, i])
        return entr

    def evaluate(self, model):
        data_inference, _ = model.encoder(torch.FloatTensor((self.metric_data["img_with_latent"]["img"])))
        data_inference = data_inference.detach().cpu().numpy()
        data_gt_latents = self.metric_data["img_with_latent"]["latent_id"]
        data_inference_discrete = self.discretize(data_inference)
        mi = self.mutual_info(
            data_inference_discrete, data_gt_latents)
        entropy = self.entropy(data_gt_latents)
        sorted_mi = np.sort(mi, axis=0)[::-1]
        mig_per_factor = np.divide(sorted_mi[0, :] - sorted_mi[1, :], entropy)
        mig_score = np.mean(mig_per_factor)

        return {"MIG_metric": mig_score,
                "MIG_metric_detail_mi": mi,
                "MIG_metric_detail_entropy": entropy,
                "mig_factor_wise":mig_per_factor}
