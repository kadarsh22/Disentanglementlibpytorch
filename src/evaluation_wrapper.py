from model_loader import get_model
from evalutation import Evaluator
from train import Trainer
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging


def run_evaluation_wrapper(configuration, data, perf_logger):
	for key, values in configuration.items():
		logging.info(' {} : {}'.format(key, values))
	Trainer.set_seed(configuration['random_seed'])
	save_config(configuration)
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	model, optimizer = get_model(configuration)
	evaluator = Evaluator(data, configuration)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")

	model, optimizer, loss = saver.load_model(model=model, optimizer=optimizer, epoch=0)
	metrics = evaluator.evaluate_model(model.eval(), epoch=0)
	z, _ = model.encoder(torch.from_numpy(data.images[0]).type(torch.FloatTensor))
	visualise_results.visualise_latent_traversal(z, model.decoder, 0)
	saver.save_results(metrics, 'metrics')
