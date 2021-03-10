import numpy as np
import os


class Teapots(object):
    def __init__(self, config):
        self.config = config
        train_data = np.load('/media/adarsh/DATA/teapots/data/images.npy',mmap_mode='r+')
        self.imgs = train_data["imgs"]
        self.latent_values = train_data["latents"]
        self.exp_name = config['experiment_name']

    def discretize(self,data, num_bins=20):

        discretized = np.zeros_like(data, dtype=np.int32)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(
                data[:, i],
                np.histogram(data[:, i], num_bins)[1][:-1])
            assert np.min(discretized[:, i]) == 1
            assert np.max(discretized[:, i]) == num_bins
            discretized[:, i] -= 1
        return discretized
