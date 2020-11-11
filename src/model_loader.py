import sys

sys.path.insert(0, './models/')
from utils import *
import torch.optim as optim
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
        model = InfoGan(config)
        # model.decoder.apply(weights_init_normal)
        # model.encoder.apply(weights_init_normal)
        d_optimizer = torch.optim.Adam([{'params': model.encoder.module_shared.parameters()},
                                        {'params': model.encoder.module_D.parameters()}],
                                       lr=2e-4, betas=(config['beta1'], config['beta2']))
        g_optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                        {'params': model.encoder.module_Q.parameters()},
                                        {'params': model.encoder.latent_disc.parameters()}],
                                       lr=1e-3, betas=(config['beta1'], config['beta2']))
        optimizer = (d_optimizer, g_optimizer)
    return model, optimizer
