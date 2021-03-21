import numpy as np
import torch


class FactorVAEMetric(object):
	""" Impementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

	def __init__(self, metric_data, device_id, config):
		super(FactorVAEMetric, self).__init__()
		self.metric_data = metric_data
		self.device_id = device_id
		self.config = config

	def evaluate(self, model):
		eval_std_inference,_ = model.encoder(torch.FloatTensor(self.metric_data["img_eval_std"]))
		eval_std_inference = eval_std_inference.detach().cpu().numpy()
		eval_std = np.std(eval_std_inference, axis=0, keepdims=True)

		labels = set(data["label"] for data in self.metric_data["groups"])

		train_data = np.zeros((eval_std.shape[1], len(labels)))

		for data in self.metric_data["groups"]:
			data_inference ,_= model.encoder(torch.FloatTensor((data["img"])))
			data_inference = data_inference.detach().cpu().numpy()
			data_inference /= eval_std
			data_std = np.std(data_inference, axis=0)
			predict = np.argmin(data_std)
			train_data[predict, data["label"]] += 1

		total_sample = np.sum(train_data)
		maxs = np.amax(train_data, axis=1)
		correct_sample = np.sum(maxs)

		correct_sample_revised = np.flip(np.sort(maxs), axis=0)
		correct_sample_revised = np.sum(
			correct_sample_revised[0: train_data.shape[1]])

		return float(correct_sample) / total_sample ,(float(correct_sample_revised) /
                                             total_sample)




