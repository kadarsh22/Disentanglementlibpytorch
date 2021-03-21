import numpy as np
import torch
import torchvision
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ioff()

teapots_metric_path = "/home/adarsh/PycharmProjects/Disentaglement/data/teapots/sliced_metric.npz"
teapots_train_path = "/home/adarsh/PycharmProjects/Disentaglement/data/teapots/sliced_train.npz"


class Teapots(object):
	def __init__(self,config):
		self.config = config
		self.exp_name = config['experiment_name']
		self.data_shape = [64, 64, 3]  # Load the data so that we can sample from it.

		self.train_images , self.metric_data ,self.train_latents = self.get_metric_data()
		self.num_factors = 5

	def get_metric_data(self):
		with open(teapots_metric_path, "rb") as data_file:
			metric_data = np.load(teapots_metric_path)


		with open(teapots_train_path, "rb") as data_file:
			train_data = np.load(teapots_train_path)
		metric_data_groups = []
		M = metric_data["imgs"].shape[0]
		for i in range(M):
			metric_data_groups.append(
				{"img": metric_data["imgs"][i],
				 "label": metric_data["latents"][i]})

		selected_ids = np.random.permutation(range(train_data["imgs"].shape[0]))
		selected_ids = selected_ids[0: int(train_data["imgs"].shape[0] / 10)]
		metric_data_eval_std = train_data["imgs"][selected_ids]

		selected_ids = np.random.permutation(range(train_data["imgs"].shape[0]))
		selected_ids = selected_ids[0: int(train_data["imgs"].shape[0] / 10)]
		random_imgs = train_data["imgs"][selected_ids]
		random_latents = train_data["latents"][selected_ids]

		random_latent_ids = self.discretize(random_latents)

		metric_data_img_with_latent = {
			"img": random_imgs,
			"latent": random_latents,
			"latent_id": random_latent_ids,
			"is_continuous": [True, True, True, True, True]}

		metric_data = {
			"groups": metric_data_groups,
			"img_eval_std": metric_data_eval_std,
			"img_with_latent": metric_data_img_with_latent}

		return train_data["imgs"], metric_data ,train_data["latents"]


	def discretize(self,data, num_bins=20):
		""" Adapted from:
			https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation
			/metrics/utils.py
		"""
		discretized = np.zeros_like(data, dtype=np.int32)
		for i in range(data.shape[1]):
			discretized[:, i] = np.digitize(
				data[:, i],
				np.histogram(data[:, i], num_bins)[1][:-1])
			assert np.min(discretized[:, i]) == 1
			assert np.max(discretized[:, i]) == num_bins
			discretized[:, i] -= 1
		return discretized, num_bins

	def show_images_grid(self, num_images=25):
		path = os.getcwd() + f'/results/{self.exp_name}' + '/visualisations/input.jpeg'
		ncols = int(np.ceil(num_images ** 0.5))
		nrows = int(np.ceil(num_images / ncols))
		_, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
		axes = axes.flatten()

		for ax_i, ax in enumerate(axes):
			if ax_i < num_images:
				ax.imshow(self.train_data['imgs'][ax_i])
				ax.set_xticks([])
				ax.set_yticks([])
			else:
				ax.axis('off')
		plt.savefig(path)


