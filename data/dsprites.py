import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import random

SCREAM_PATH = "/home/adarsh/Documents/data/scream/scream.jpg"
dsprites_path = "/home/adarsh/PycharmProjects/Disentaglement/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64" \
				".npz"


class DSprites(object):
	def __init__(self, config):
		self.config = config
		self.exp_name = config['experiment_name']
		self.data_shape = [64, 64, 1]  # Load the data so that we can sample from it.
		with open(dsprites_path, "rb") as data_file:
			# Data was saved originally using python2, so we need to set the encoding.
			data = np.load(dsprites_path, encoding="latin1", allow_pickle=True)

		metadata = data['metadata'][()]
		self.latents_sizes = metadata['latents_sizes']

		# if full data load the entire dataset else only load images corresponding to one shape
		if config['full_data']:
			self.images = np.array(data["imgs"])
			self.latents_values = data['latents_values']
			self.latents_classes = data['latents_classes']
		else:
			self.images = np.array(data["imgs"])[:32 * 32 * 40 * 6]
			self.latents_values = data['latents_values'][:32 * 32 * 40 * 6]
			self.latents_classes = data['latents_classes'][:32 * 32 * 40 * 6]
			self.latents_sizes[1] = 1

		self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
		self.num_factors = 6
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

	def latent_to_index(self, latents):
		return np.dot(latents, self.latents_bases).astype(int)

	def sample_latent_values(self, latents_sampled):

		indices_sampled = self.latent_to_index(latents_sampled)
		# imgs_sampled = self.images[indices_sampled]
		latent_values = self.latents_values[indices_sampled]
		return latent_values


	def sample_oracle_training_data(self, num_training_samples):

		latent_reference = self.sample_latent(size=num_training_samples)
		latent_copy = np.copy(latent_reference)
		probability_vec = torch.FloatTensor([0.5, 0.5, 0.5 , 0.5 ,0.5]*num_training_samples)
		labels_array = torch.bernoulli(probability_vec).view(-1,5)
		y_pos_template = torch.from_numpy(self.sample_images_from_latent(self.latents_classes[[x for x in range(32)]]))
		x_pos_template = torch.from_numpy(self.sample_images_from_latent(self.latents_classes[[0] + [x for x in range(32, 32 * 32, 32)]]))
		orientation_template =  torch.from_numpy(self.sample_images_from_latent(self.latents_classes[[0] + [x for x in range(32 * 32 + 1, 40 * 32 * 32 + 1, 32 * 32)]]))
		size_template = torch.from_numpy(self.sample_images_from_latent(self.latents_classes[[0] + [x for x in range(40 * 32 * 32 + 1,6 * 40 * 32 * 32 + 1, 40 * 32 * 32)]]))
		shape_template = torch.from_numpy(self.sample_images_from_latent(self.latents_classes[[0] +[x for x in range(6 * 40 * 32 * 32 + 1, 3 * 6 * 40 * 32 * 32 + 1, 6 * 40 * 32 * 32)]]))

		for i in range(1,6):
			latent_unit =  latent_reference[:,i]
			change_index = labels_array[:,i-1]
			replace_list = []
			for current_value in latent_unit[change_index>0]:
				if i == 1: ##shape
					replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) +1, 3)))
				if i == 2: # size
					replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) +1, 6)))
				if i == 3: # orientation
					replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) +1, 40)))
				if i == 4: # xposition
					replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) +1, 32)))
				if i == 5: # yposition
					replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) +1, 32)))
				replace_list.append(replace_value)
			latent_copy[change_index>0, i] = np.array(replace_list)
		shape_ = torch.stack([shape_template[int(i)] for i in latent_copy[:,1]]).view(-1,1,64,64)
		size_ = torch.stack([size_template[int(i)] for i in latent_copy[:,2]]).view(-1,1,64,64)
		orientation_ = torch.stack([orientation_template[int(i)] for i in latent_copy[:,3]]).view(-1,1,64,64)
		x_pos_ = torch.stack([x_pos_template[int(i)] for i in latent_copy[:,4]]).view(-1,1,64,64)
		y_pos_ = torch.stack([y_pos_template[int(i)] for i in latent_copy[:,5]]).view(-1,1,64,64)
		ref_imgs = torch.Tensor(self.sample_images_from_latent(latent_reference)).view(-1,1,64,64)
		training_images = torch.cat((ref_imgs,shape_,size_,orientation_,x_pos_,y_pos_),dim= 1)
		return training_images , torch.FloatTensor(labels_array)