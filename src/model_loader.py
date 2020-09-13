import sys

sys.path.insert(0, './models/')
from utils import *
import itertools


def get_model(config):
    device = torch.device('cuda:' + str(config['device_id']))
    model_name = config['model_name']
    if model_name == 'beta_vae':
        from beta_vae import VAE
        loss = None
        model = VAE().to(device)
        optimizer = (torch.optim.Adagrad(model.parameters(), lr=config['learning_r']),)
    elif model_name == 'factor_vae':
        raise NotImplementedError
    elif model_name == 'betavae_cnn':
        from beta_vae_cnn import VAE
        loss = None
        model = VAE(latent_dim=config['latent_dim']).to(device)
        optimizer = (torch.optim.Adam(model.parameters(), lr=config['learning_r']),)
    elif model_name == 'infogan':
        from infogan import InfoGan
        model = InfoGan(config)
        loss =None
        model.decoder.apply(weights_init_normal)
        model.encoder.apply(weights_init_normal)
        model.cr_disc.apply(weights_init_normal)
        d_optimizer = torch.optim.Adam([{'params': model.encoder.conv1.parameters()},
                                        {'params': model.encoder.conv2.parameters()},
                                        {'params': model.encoder.conv3.parameters()},
                                        {'params': model.encoder.conv4.parameters()},
                                        {'params': model.encoder.linear1.parameters()},
                                        {'params': model.encoder.linear2.parameters()},
                                        {'params': model.encoder.linear3.parameters()}],
                                       lr=config['learning_r_D'], betas=(config['beta1'], config['beta2']))
        g_optimizer = torch.optim.Adam([{'params': model.decoder.parameters()}],
                                       lr=config['learning_r_G'], betas=(config['beta1'], config['beta2']))
        cr_optimizer = torch.optim.Adam([{'params': model.cr_disc.parameters()}],
                                       lr=config['learning_r_CR'], betas=(config['beta1'], config['beta2']))

        optimizer = (d_optimizer, g_optimizer,cr_optimizer)

    elif model_name == 'deepinfomax':
        from deep_infomax import InfoMax
        model = InfoMax()
        encoder_optim = torch.optim.Adam(model.encoder.parameters(), lr=1e-4)
        global_loss_optim = torch.optim.Adam(model.global_discriminator.parameters(), lr=1e-4)
        local_loss_optim = torch.optim.Adam(model.local_discriminator.parameters(), lr=1e-4)
        prior_loss_optim = torch.optim.Adam(model.prior.parameters(), lr=1e-4)
        return model , (encoder_optim, global_loss_optim, local_loss_optim, prior_loss_optim)

    elif model_name == 'cnn':
        from classifier import Classifier
        model = Classifier()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        loss = None
    return model, optimizer
