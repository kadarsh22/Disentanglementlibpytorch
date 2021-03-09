import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt

path = "/home/adarsh/PycharmProjects/Disentaglement/data/3dshapes.h5"
import h5py
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}

class shapes3d(object):
	def __init__(self, config):
		self.config = config
		self.exp_name = config['experiment_name']
		self.data_shape = [64, 64, 3]  # Load the data so that we can sample from it.
		dataset = h5py.File(path, 'r')
		self.images = dataset['images']
		self.labels = dataset['labels']
		self.image_shape = self.images.shape[1:]  # [64,64,3]
		self.label_shape = self.labels.shape[1:]  # [6]
		self.n_samples = self.labels.shape[0]

	def get_index(self,factors):
		""" Converts factors to indices in range(num_data)
		Args:
		  factors: np array shape [6,batch_size].
				   factors[i]=factors[i,:] takes integer values in
				   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

		Returns:
		  indices: np array shape [batch_size].
		"""
		indices = 0
		base = 1
		for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
			indices += factors[factor] * base
			base *= _NUM_VALUES_PER_FACTOR[name]
		return indices

	def sample_random_batch(self,batch_size):
		""" Samples a random batch of images.
		Args:
		  batch_size: number of images to sample.

		Returns:
		  batch: images shape [batch_size,64,64,3].
		"""
		indices = np.random.choice(self.n_samples, batch_size)
		ims = []
		for ind in indices:
			im = self.images[ind]
			im = np.asarray(im)
			ims.append(im)
		ims = np.stack(ims, axis=0)
		ims = ims / 255.  # normalise values to range [0,1]
		ims = ims.astype(np.float32)
		return ims.reshape([batch_size, 64, 64, 3])

	def sample_batch(self,batch_size, fixed_factor, fixed_factor_value):
		""" Samples a batch of images with fixed_factor=fixed_factor_value, but with
			the other factors varying randomly.
		Args:
		  batch_size: number of images to sample.
		  fixed_factor: index of factor that is fixed in range(6).
		  fixed_factor_value: integer value of factor that is fixed
			in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

		Returns:
		  batch: images shape [batch_size,64,64,3]
		"""
		factors = np.zeros([len(_FACTORS_IN_ORDER), batch_size],
						   dtype=np.int32)
		for factor, name in enumerate(_FACTORS_IN_ORDER):
			num_choices = _NUM_VALUES_PER_FACTOR[name]
			factors[factor] = np.random.choice(num_choices, batch_size)
		factors[fixed_factor] = fixed_factor_value
		indices = self.get_index(factors)
		ims = []
		for ind in indices:
			im = self.images[ind]
			im = np.asarray(im)
			ims.append(im)
		ims = np.stack(ims, axis=0)
		ims = ims / 255.  # normalise values to range [0,1]
		ims = ims.astype(np.float32)
		return ims.reshape([batch_size, 64, 64, 3])



