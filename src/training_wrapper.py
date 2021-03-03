from model_loader import get_model
from train import Trainer
from evalutation import Evaluator
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging
from utils import print_network


def run_training_wrapper(configuration, data, perf_logger):
	for key, values in configuration.items():
		logging.info(' {} : {}'.format(key, values))
	save_config(configuration)
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	model, optimizer = get_model(configuration)
	model_trainer = Trainer(data, configuration)
	evaluator = Evaluator(data, configuration)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")
	print_network(model.encoder)
	print_network(model.decoder)
	cr_optimizer = optimizer[2]
	model, optimizer, loss = saver.load_model(model=model, optimizer=optimizer)
	optimizer_list = list(optimizer)
	optimizer_list.append(cr_optimizer)
	optimizer = tuple(optimizer_list)
	model.encoder.cuda()
	model.decoder.cuda()
	model.cr_disc.cuda()
	for i in range(configuration['epochs']):
		if configuration['model_arch'] == 'vae':
			model.train()
			model, loss, optimizer = model_trainer.train_vae(model, optimizer[0], i)
		elif configuration['model_arch'] == 'gan':
			model.encoder.train()
			model.decoder.train()
			model, optimizer, loss = model_trainer.train_classifier(model)
			# model, loss, optimizer = model_trainer.train_gan(model, optimizer, i)
		elif configuration['model_arch'] == 'cnn':
			model.train()
			model, optimizer, loss = model_trainer.train_classifier(model, optimizer, i)

		if i % configuration['logging_freq'] == 0 and i != 0:
			if configuration['model_arch'] == 'vae':
				model.eval()
			else:
				model.encoder.eval()
				model.decoder.eval()
			metrics = evaluator.evaluate_model(model, i)
			z, _ = model.encoder(torch.from_numpy(data.images[0]).type(torch.FloatTensor))
			perf_logger.start_monitoring("Latent Traversal Visualisations")
			visualise_results.visualise_latent_traversal(z, model.decoder, i)
			perf_logger.stop_monitoring("Latent Traversal Visualisations")

	perf_logger.start_monitoring("Saving Results")
	saver.save_results(metrics, 'metrics')
	saver.save_results(loss, 'loss')
	perf_logger.stop_monitoring("Saving Results")
	perf_logger.start_monitoring("Saving plots")
	visualise_results.generate_plot_save_results(metrics, 'metrics')
	visualise_results.generate_plot_save_results(loss, 'loss')
	return loss, metrics
