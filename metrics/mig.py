import numpy as np
import logging
from sklearn.metrics import mutual_info_score
import torch

class MIG(object):
    """
            Implementation of the metric in: MIG
    """
    def __init__(self, dsprites,device_id):
        super(MIG, self).__init__()
        self.data = dsprites
        self.device_id = device_id


    def compute_mig(self, model, num_train, random_state, batch_size=128):
        # logging.info("Generating training set.")
        representations, ground_truth = self.generate_batch_factor_code(model, num_train, random_state, batch_size)
#        representations = ground_truth             ## used for testing
        assert representations.shape[1] == num_train
        return self._compute_mig(representations, ground_truth)

    def _compute_mig(self,representations, ground_truth):
        """Computes score based on both training and testing codes and factors."""
        score_dict = {}
        discretized_representations = self._histogram_discretize(representations)
        m = self.discrete_mutual_info(discretized_representations, ground_truth)
        assert m.shape[0] == representations.shape[0]
        assert m.shape[1] == ground_truth.shape[0]
        # m is [num_latents, num_factors]
        entropy = self.discrete_entropy(ground_truth)
        sorted_m = np.sort(m, axis=0)[::-1]
        dimension_wise_mig  = np.divide((sorted_m[0, :] - sorted_m[1, :])[1:], entropy[1:])
        if np.isnan(np.min(dimension_wise_mig)):
            logging.info("zeros found while computing MIG")
            dimension_wise_mig = np.nan_to_num(dimension_wise_mig, copy=True, nan=1.0, posinf=None, neginf=None)
        score_dict["discrete_mig"] = np.mean(dimension_wise_mig)
        return score_dict

    def _histogram_discretize(self , target, num_bins=40):
        """Discretization based on histograms."""
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(target[i, :], np.histogram(
                target[i, :], num_bins)[1][:-1])
        return discretized

    def generate_batch_factor_code(self, model,
                                   num_points, random_state, batch_size):

        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_factors = self.data.sample_latent(num_points_iter)
            current_observations = torch.from_numpy(self.data.sample_images_from_latent(current_factors))
            current_representations , _ = model.encoder(current_observations)
            current_representations = current_representations.data.cpu()
            if i == 0:
                factors = current_factors
                representations = current_representations
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, current_representations))
            i += num_points_iter
        return np.transpose(representations), np.transpose(factors)


    def discrete_entropy(self, ys):
        """Compute discrete mutual information."""
        num_factors = ys.shape[0]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = mutual_info_score(ys[j, :], ys[j, :])
        return h

    def discrete_mutual_info(self, mus, ys):
        """Compute discrete mutual information."""
        num_codes = mus.shape[0]
        num_factors = ys.shape[0]
        m = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):
                m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
        return m
