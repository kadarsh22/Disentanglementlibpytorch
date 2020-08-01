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
parser.add_argument('--model_arch', type=str, default='vae', choices=['vae', 'gan', 'deepinfomax'],
					help='architecture of model')
parser.add_argument('--model_name', type=str, default='beta_vae', choices=['beta_vae', 'factor_vae', 'infogan',
																		   'betavae_cnn'], help='architecture of '
																								'model')
parser.add_argument('--dataset', type=str, default='dsprites', choices=['celeba', 'noisydsprites', 'coloredsprites',
																		'cars3d'], help='name of the dataset')
parser.add_argument('--epochs', type=int, default=12, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=2048, help='The size of batch')
parser.add_argument('--device_id', type=int, default=0, help='Device id of gpu')
parser.add_argument('--gan_type', type=str, default='infoGAN', choices=['dcgan', 'infogan'], help='The type of GAN')
parser.add_argument('--random_seed', type=int, default=123, help='Random seeds to run for ')
parser.add_argument('--latent_dim', type=int, default=10, help='Number of latent units ')
parser.add_argument('--learning_r', type=float, default=5e-4, help='Number of latent units ')
parser.add_argument('--logging_freq', type=int, default=5, help='Frequency at which result  should be logged')
parser.add_argument('--full_data', type=bool, default=True, help='whether to use full data or not')


def get_config(inputs):
	config = parser.parse_args(inputs)
	return config


def save_config(config):
	cwd = os.getcwd() + f'/results/{config.experiment_name}'  # project root
	models_dir = cwd + '/models'  # models directory
	visualisations_dir = cwd + '/visualisations'  # directory in which images and plots are saved
	os.makedirs(cwd, exist_ok=True)
	os.makedirs(models_dir, exist_ok=True)
	os.makedirs(visualisations_dir, exist_ok=True)
	with open(f'{cwd}/config.json', 'w') as fp:
		json.dump(config.__dict__, fp, indent=4, sort_keys=True)
	return
