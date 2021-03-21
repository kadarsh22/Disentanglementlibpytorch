import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class BetaVAEMetric(object):
    """ Impementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

    def __init__(self, metric_data, device_id, config):
        super(BetaVAEMetric, self).__init__()
        self.metric_data = metric_data
        self.device_id = device_id
        self.config = config

    def evaluate(self, model):
        features = []
        labels = []

        for data in self.metric_data["groups"]:
            data_inference,_= model.encoder(torch.FloatTensor(data["img"]))
            data_inference = data_inference.detach().cpu().numpy()
            data_diff = np.abs(data_inference[0::2] - data_inference[1::2])
            data_diff_mean = np.mean(data_diff, axis=0)
            features.append(data_diff_mean)
            labels.append(data["label"])

        features = np.vstack(features)
        labels = np.asarray(labels)

        classifier =  LogisticRegression()
        classifier.fit(features, labels)

        acc = classifier.score(features, labels)
        return acc


