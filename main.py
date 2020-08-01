import sys
import torch
from config import save_config
from config import get_config
from data_loader import get_data_loader
from model_loader import get_model
from train import Trainer
from evalutation import Evaluator
from saver import Saver
from visualiser import Visualiser
from logger import PerfomanceLogger
import logging


def main(configuration):
	perf_logger = PerfomanceLogger()
	perf_logger.start_monitoring("Saving configurations")
	for arg in vars(configuration):
		logging.info(' {} : {}'.format(arg, getattr(configuration, arg)))
	save_config(configuration)
	perf_logger.stop_monitoring("Saving configurations")
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	data = get_data_loader(configuration)
	model, optimizer = get_model(configuration)
	model_trainer = Trainer(data, configuration)
	evaluator = Evaluator(data, configuration)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")

	for i in range(configuration.epochs):
		model.train()
		model, loss = model_trainer.train_vae(model, optimizer, i)
		if i % configuration.logging_freq == 0 and i != 0:
			perf_logger.start_monitoring("Saving Model")
			saver.save_model((model,), (optimizer,), loss, epoch=i)
			perf_logger.stop_monitoring("Saving Model")
			model.eval()
			metrics = evaluator.evaluate_model(model, i)
			z, _ = model.encoder(torch.from_numpy(data.images[0]).type(torch.FloatTensor))
			perf_logger.start_monitoring("Latent Traversal Visualisations")
			visualise_results.visualise_latent_traversal(z, model.decoder, 2 / 3, i)
			perf_logger.stop_monitoring("Latent Traversal Visualisations")

	perf_logger.start_monitoring("Saving Results")
	saver.save_results(metrics, 'metrics')
	saver.save_results(loss, 'loss')
	perf_logger.stop_monitoring("Saving Results")
	perf_logger.start_monitoring(("Saving plots"))
	visualise_results.generate_plot_save_results(metrics, 'metrics')
	visualise_results.generate_plot_save_results(loss, 'loss')


if __name__ == "__main__":
	config = get_config(sys.argv[1:])
	Trainer.set_seed(config.random_seeds)
	PerfomanceLogger.configure_logger(config)
	main(config)
