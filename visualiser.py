import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import itertools



class Visualiser(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = config['experiment_name']

	def generate_plot_save_results(self, results, plot_type):
		file_location = os.getcwd() + f'/results/{self.experiment_name}' + '/visualisations/plots/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		plt.figure()
		for name, values in results.items():
			x_axis = list(range(len(values)))
			plt.plot(x_axis, values, label=name)
		plt.legend(loc="upper right")
		path = file_location + str(plot_type) + '.jpeg'
		plt.savefig(path)

	def visualise_latent_traversal(self, initial_rep, decoder, interval, epoch_num):
		interpolation = torch.arange(-3, 3 + 0.1, interval)
		rep_org = initial_rep
		file_location = os.getcwd() + f'/results/{self.experiment_name}' + '/visualisations/latent_traversal/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(epoch_num) + '.jpeg'
		samples = []
		for j in range(self.config['latent_dim']):
			temp = initial_rep.data[:, j].clone()
			for k in interpolation:
				rep_org.data[:, j] = k
				sample = torch.sigmoid(decoder(rep_org))  # TODO need not be sigmoid
				sample = sample.view(-1, 64, 64)
				samples.append(sample)
			rep_org.data[:, j] = temp
		grid_img = torchvision.utils.make_grid(samples, nrow=10, padding=10, pad_value=1)
		grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
		plt.imsave(path, grid.data.numpy())

	def visualise_ablation_results(self, y_axis, x_axis, plot_name, legend_list):
		file_location = os.getcwd() + f'/results/{self.experiment_name}/'
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


