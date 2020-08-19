import sys

sys.path.insert(0, './models/')
import itertools
from utils import *


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
        model.decoder.apply(weights_init_normal)
        model.encoder.apply(weights_init_normal)
        g_optimizer = set_optimizer([model.decoder.parameters(), model.encoder.module_Q.parameters(
        ), model.encoder.latent_cont.parameters()], lr=config['learning_r_G'], config=config)
        d_optimizer = set_optimizer([model.encoder.module_shared.parameters(), model.encoder.module_D.parameters()],
                                    lr=config['learning_r_D'], config=config)
        optimizer = (d_optimizer, g_optimizer)
    return model, optimizer


def set_optimizer(param_list, lr, config):
    params_to_optimize = itertools.chain(*param_list)
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, betas=(config['beta1'], config['beta2']))
    return optimizer
