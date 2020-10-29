import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt

SCREAM_PATH = "/home/adarsh/Documents/data/scream/scream.jpg"
cdsprites_path = "/home/adarsh/PycharmProjects/Disentaglement/data/cdsprites/cdsprites_ndarray_64x64.npz"


class CircularDSprites(object):
	def __init__(self, config):
		self.config = config
		self.exp_name = config['experiment_name']
		self.data_shape = [64, 64, 1]  # Load the data so that we can sample from it.
		with open(cdsprites_path, "rb") as data_file:
			# Data was saved originally using python2, so we need to set the encoding.
			data = np.load(cdsprites_path, encoding="latin1", allow_pickle=True)

		metadata = data['metadata'][()]
		self.latents_sizes = metadata['latents_sizes']
		self.images = np.array(data["imgs"]).reshape(-1, 64, 64, 1).astype(np.float)
		self.latents_values = data['latents_values']
		# self.latents_classes = data['latents_classes']


		self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
		self.num_factors = 2
		# self.show_images_grid()

	def show_images_grid(self, nrows=10):
		path = os.getcwd() + f'/results/{self.exp_name}' + '/visualisations/input.jpeg'
		index = np.random.choice(self.images.shape[0], nrows * nrows, replace=False)
		batch_tensor = torch.from_numpy(self.images[index])
		grid_img = torchvision.utils.make_grid(batch_tensor.view(-1, 1, 64, 64), nrow=10, padding=5, pad_value=1)
		grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
		plt.imsave(path, grid.numpy())

	def sample(self, num):
		latents_sampled = self.sample_latent(size=num)
		indices_sampled = self.latent_to_index(latents_sampled)
		imgs_sampled = self.images[indices_sampled]
		latents_sampled = self.latents_values[indices_sampled]
		return latents_sampled, imgs_sampled

	def sample_latent(self, size=1):
		"""
		Generate a vector with size of ground truth factors and random fill each column with values from range
		:param size:
		:return: latents
		"""

		samples = np.zeros((size, self.latents_sizes.size))
		for lat_i, lat_size in enumerate(self.latents_sizes):
			samples[:, lat_i] = np.random.randint(lat_size, size=size)

		return samples

	def sample_images_from_latent(self, latent):
		indices_sampled = self.latent_to_index(latent)
		imgs_sampled = self.images[indices_sampled]
		return imgs_sampled

	def sample_images_from_index(self, index):
		imgs_sampled = self.images[index]
		return imgs_sampled

	def latent_to_index(self, latents):
		return np.dot(latents, self.latents_bases).astype(int)

	def sample_latent_values(self, latents_sampled):

		indices_sampled = self.latent_to_index(latents_sampled)
		# imgs_sampled = self.images[indices_sampled]
		latent_values = self.latents_values[indices_sampled]
		return latent_values