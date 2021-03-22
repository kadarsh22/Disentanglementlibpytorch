import os
import torch
import torchvision
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.ioff()
import itertools



class Visualiser(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = config['experiment_name']

	def show_images_grid(self, save_path, samples, num_images=25):
		# ncols = int(np.ceil(num_images ** 0.5))
		# nrows = int(np.ceil(num_images / ncols))
		ncols = 5
		nrows = 10
		_, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
		axes = axes.flatten()

		for ax_i, ax in enumerate(axes):
			if ax_i < num_images:
				ax.imshow(samples[ax_i].permute(1,2,0).cpu().detach().numpy())
				ax.set_xticks([])
				ax.set_yticks([])
			else:
				ax.axis('off')
		plt.savefig(save_path)

	def generate_plot_save_results(self, results, plot_type):
		file_location = os.path.dirname(os.getcwd())+ f'/results/{self.experiment_name}' + '/visualisations/plots/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		plt.figure()
		for name, values in results.items():
			x_axis = [self.config['logging_freq'] * i for i in range(len(values))]
			plt.plot(x_axis, values, label=name)
		plt.legend(loc="upper right")
		path = file_location + str(plot_type) + '.jpeg'
		plt.savefig(path)

	def visualise_latent_traversal(self, initial_rep, decoder, epoch_num):
		interval_start = self.config['interval_start']
		interval = (2*(interval_start))/10
		interpolation = torch.arange(-1*interval_start, interval_start, interval)
		rep_org = initial_rep
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/visualisations/latent_traversal/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(epoch_num) + '.jpeg'
		samples = []
		z_ = torch.rand((1, self.config['noise_dim'])).cuda()
		for j in range(self.config['latent_dim']):
			temp = initial_rep.data[:, j].clone()
			for k in interpolation:
				rep_org.data[:, j] = k
				if self.config['model_arch'] == 'gan':
					final_rep = torch.cat((z_, rep_org), dim=1)
					sample = decoder(final_rep)  # TODO need not be sigmoid
				else:
					sample = torch.sigmoid(decoder(rep_org))
				sample = sample.view(-1, 64, 64)
				samples.append(sample)
			rep_org.data[:, j] = temp
		self.show_images_grid(path, samples, num_images=len(samples))

	def visualise_ablation_results(self, y_axis, x_axis, plot_name, legend_list):
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}/'
		plt.figure()
		for x, y, legend in zip(x_axis, y_axis, legend_list):
			plt.plot(x, y, label=str(legend))
		plt.legend(loc="upper right")
		plt.savefig(file_location + str(plot_name) + '.jpeg')

	def visualise_4d_ablation_plots(self, param_one, param_two, values, z_label, x_label, y_label):
		import matplotlib
		matplotlib.use('GTK3Agg')
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		x = param_one
		y = param_two
		z = values
		c = [x for x in range(len(values[0]))]
		combo = list(itertools.product(x, y))
		for i,j in zip(combo,z):
			# img = ax.scatter([i[0]]*len(c), [i[1]]*len(c), j, c=c, cmap=plt.hot())
			ax.plot3D([i[0]]*len(c),[i[1]]*len(c), j)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_zlabel(z_label)
		# fig.colorbar(img)
		plt.show()
