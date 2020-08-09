import argparse
import json
import os
import sys

experiment_name = input("Enter experiment name ")
if experiment_name == '':
	print('enter valid experiment name')
	sys.exit()

experiment_description = input("Enter description of experiment ")
if experiment_description == '':
	print('enter proper description')
	sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=experiment_name)
parser.add_argument('--experiment_description', type=str, default=experiment_description)

## general configuration
parser.add_argument('--model_arch', type=str, default='gan', choices=['vae', 'gan', 'deepinfomax'],
					help='architecture of model')
parser.add_argument('--model_name', type=str, default='infogan', choices=['beta_vae', 'factor_vae', 'infogan',
																		   'betavae_cnn'], help='architecture of '
																								'model')
parser.add_argument('--dataset', type=str, default='dsprites', choices=['celeba', 'noisydsprites', 'coloredsprites',
																		'cars3d'], help='name of the dataset')
parser.add_argument('--epochs', type=int, default=30, help='The number of epochs to run')
parser.add_argument('--logging_freq', type=int, default=5, help='Frequency at which result  should be logged')
parser.add_argument('--full_data', type=bool, default=False, help='whether to use full data or not')
parser.add_argument('--ablation', type=bool, default=False, help='wether to run in ablation study mode or not')
parser.add_argument('--device_id', type=int, default=0, help='Device id of gpu')
parser.add_argument('--random_seed', type=int, default=123, help='Random seeds to run for ')

## VAE configurations
parser.add_argument('--batch_size', type=int, default=1024, help='The size of batch')
parser.add_argument('--latent_dim', type=int, default=5, help='Number of latent units ')
parser.add_argument('--learning_r', type=float, default=5e-4, help='Number of latent units ')

# GAN configurations
parser.add_argument('--noise_dim', type=int, default=5, help='Number of noise_dim')
parser.add_argument('--learning_r_G', type=float, default=0.001, help='learning rate of generator')
parser.add_argument('--learning_r_D', type=float, default=0.0002, help='learning rate of discriminator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 optimizer')

# parser.add_argument('--random_seeds', type=list, default=[123,34,134], help='Random seeds to run for ')
# parser.add_argument('--latent_dims', type=list, default=[5,10], help='Number of latent units ')
# parser.add_argument('--learning_rs', type=list, default=[5e-4,1e-3], help='Number of latent units ')


def get_config(inputs):
	config = parser.parse_args(inputs)
	return config.__dict__


def save_config(config):
	exp_name = config['experiment_name']
	cwd = os.getcwd() + f'/results/{exp_name}'  # project root
	models_dir = cwd + '/models'  # models directory
	visualisations_dir = cwd + '/visualisations'  # directory in which images and plots are saved
	os.makedirs(cwd, exist_ok=True)
	os.makedirs(models_dir, exist_ok=True)
	os.makedirs(visualisations_dir, exist_ok=True)
	with open(f'{cwd}/config.json', 'w') as fp:
		json.dump(config, fp, indent=4, sort_keys=True)
	return

def str2bool(v):
	return v.lower() in ('true', '1')