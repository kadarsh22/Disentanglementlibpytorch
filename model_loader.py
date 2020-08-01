import sys

sys.path.insert(0, './models/')
import torch


def get_model(config):
	device = torch.device('cuda:' + str(config.device_id))
	model_name = config.model_name
	if model_name == 'beta_vae':
		from beta_vae import VAE
		model = VAE().to(device)
		optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_r)
	elif model_name == 'factor_vae':
		raise NotImplementedError
	elif model_name == 'infogan':
		raise NotImplementedError
	elif model_name == 'betavae_cnn':
		from beta_vae_cnn import VAE
		model = VAE(latent_dim=config.latent_dim).to()
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_r)
	return model, optimizer
