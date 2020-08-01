import os
import torch


class Saver(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = self.config['experiment_name']

	def save_model(self, model, optimizer, loss, epoch=0):
		cwd = os.getcwd() + f'/results/{self.experiment_name}'  # project root
		models_dir = cwd + '/models/'

		if not os.path.exists(models_dir):
			os.makedirs(models_dir)

		if self.config['model_arch'] == 'vae':
			torch.save({
				'epoch': epoch,
				'model_state_dict': model[0].state_dict(),
				'optimizer_state_dict': optimizer[0].state_dict(),
				'loss': loss
			}, os.path.join(models_dir, str(epoch) + '_vae.pkl'))
			torch.save(model[0].state_dict(), os.path.join(models_dir, '_vae.pkl'))
		elif self.config['model_arch'] == 'gan':
			torch.save({
				'epoch': epoch,
				'gen_state_dict': model[0].state_dict(),
				'dis_state_dict': model[1].state_dict(),
				'gen_optimizer_state_dict': optimizer[0].state_dict(),
				'dis_optimizer_state_dict': optimizer[1].state_dict(),
				'loss': loss
			}, os.path.join(models_dir, str(epoch) + '_gan.pkl'))
		else:
			raise NotImplementedError

	def load_model(self, model, optimizer, epoch):
		cwd = os.getcwd() + f'/results/{self.experiment_name}'  # project root
		models_dir = cwd + '/models/'

		if self.config.model_arch == 'vae':
			checkpoint = torch.load(os.path.join(models_dir, str(epoch) + '_vae.pkl'))
			model[0].load_state_dict(checkpoint['model_state_dict'])
			optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])
			loss = checkpoint['loss']
			return (model[0],), (optimizer[0],), loss
		elif self.config.model_arch == 'gan':
			checkpoint = torch.load(os.path.join(models_dir, str(epoch) + '_gan.pkl'))
			model[0].load_state_dict(checkpoint['gen_state_dict'])
			model[1].load_state_dict(checkpoint['dis_state_dict'])
			optimizer[0].load_state_dict(checkpoint['gen_optimizer_state_dict'])
			optimizer[1].load_state_dict(checkpoint['dis_optimizer_state_dict'])
			loss = checkpoint['loss']
			return (model[0], model[1]), (optimizer[0], optimizer[1]), loss
		else:
			raise NotImplementedError

	def save_results(self, results, filename):
		file_location = os.getcwd() + f'/results/{self.experiment_name}' + '/experimental_results/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(filename) + '.pkl'
		torch.save(results, path)

	def load_results(self, filename):
		# noinspection PyCompatibility
		file_location = os.getcwd() + f'/results/{self.experiment_name}' + '/experimental_results/'
		path = file_location + str(filename) + '.pkl'
		results = torch.load(path)
		return results
