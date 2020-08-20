import os
import torch


class Saver(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = self.config['experiment_name']
		self.model_name = self.config['file_name']

	def save_model(self, model, optimizer, loss, epoch=0):
		cwd = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}'  # project root
		models_dir = cwd + '/models/'

		if not os.path.exists(models_dir):
			os.makedirs(models_dir)

		if self.config['model_arch'] == 'vae':
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer[0].state_dict(),
				'loss': loss
			}, os.path.join(models_dir, str(epoch) + '_vae.pkl'))
		elif self.config['model_arch'] == 'gan':
			torch.save({
				'epoch': epoch,
				'gen_state_dict': model.decoder.state_dict(),
				'dis_state_dict': model.encoder.state_dict(),
				'gen_optimizer_state_dict': optimizer[0].state_dict(),
				'dis_optimizer_state_dict': optimizer[1].state_dict(),
				'loss': loss
			}, os.path.join(models_dir, str(epoch) + '_gan.pkl'))
		else:
			raise NotImplementedError

	def load_model(self, model, optimizer):
		models_dir = os.path.dirname(os.getcwd()) + f'/pretrained_models/{self.model_name}'  # project root
		checkpoint = torch.load(models_dir)

		if self.config['model_arch'] == 'vae':
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])
			loss = checkpoint['loss']
			return model, optimizer, loss
		elif self.config['model_arch'] == 'gan':
			model.encoder.load_state_dict(checkpoint['dis_state_dict'])
			model.decoder.load_state_dict(checkpoint['gen_state_dict'])
			optimizer[0].load_state_dict(checkpoint['gen_optimizer_state_dict'])
			optimizer[1].load_state_dict(checkpoint['dis_optimizer_state_dict'])
			loss = checkpoint['loss']
			return model, (optimizer[0], optimizer[1]), loss
		else:
			raise NotImplementedError

	def save_results(self, results, filename):
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/experimental_results/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(filename) + '.pkl'
		torch.save(results, path)

	def load_results(self, filename):
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/experimental_results/'
		path = file_location + str(filename) + '.pkl'
		results = torch.load(path)
		return results
