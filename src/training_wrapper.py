from model_loader import get_model
from train import Trainer
from evalutation import Evaluator
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging


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

	for i in range(configuration['epochs']):
		if configuration['model_arch'] == 'vae':
			model.train()
			model, loss, optimizer = model_trainer.train_vae(model, optimizer[0], i)
		else:
			model.encoder.train()
			model.decoder.train()
			model, loss, optimizer = model_trainer.train_gan(model, optimizer, i)

		if i % configuration['saving_freq'] == 0 and i != 0:
			perf_logger.start_monitoring("Saving Model")
			saver.save_model(model, optimizer, loss, epoch=i)
			perf_logger.stop_monitoring("Saving Model")

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
