import sys

sys.path.insert(0, './models/')
import torch
import itertools


def get_model(config):
	device = torch.device('cuda:' + str(config['device_id']))
	model_name = config['model_name']
	if model_name == 'beta_vae':
		from beta_vae import VAE
		model = VAE().to(device)
		optimizer = (torch.optim.Adagrad(model.parameters(), lr=config['learning_r']),)
	elif model_name == 'factor_vae':
		raise NotImplementedError
	elif model_name == 'betavae_cnn':
		from beta_vae_cnn import VAE
		model = VAE(latent_dim=config['latent_dim']).to(device)
		optimizer = (torch.optim.Adam(model.parameters(), lr=config['learning_r']),)
	elif model_name == 'infogan':
		from infogan import InfoGan
		model = InfoGan(latent_dim = config['latent_dim'],noise_dim = config['noise_dim'])
		d_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['learning_r_G'], betas=(config['beta1'], config['beta2']))
		g_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=config['learning_r_D'],  betas=(config['beta1'], config['beta2']))
		info_optimizer = torch.optim.Adam(itertools.chain(model.decoder.parameters(), model.encoder.parameters()), lr=config['learning_r_G'],
									   betas=(config['beta1'], config['beta2']))
		optimizer = (d_optimizer,g_optimizer,info_optimizer)
	return model, optimizer
