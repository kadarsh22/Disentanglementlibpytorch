import sys
from config import get_config
from data_loader import get_data_loader
from train import Trainer
from training_wrapper import run_training_wrapper
from ablation_wrapper import run_ablation_wrapper
from evaluation_wrapper import run_evaluation_wrapper
from logger import PerfomanceLogger
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

def main(configuration):
	if configuration['dataset'] == 'dsprites' or 'scream_dsprites':
		configuration['low_factor_vae'] = 1
		configuration['low_beta_vae'] = 1
		configuration['mig_entropy_start'] = 1
		configuration['mig_start'] = 0
	else:
		configuration['low_factor_vae'] = 0
		configuration['low_beta_vae'] = 0
		configuration['mig_entropy_start'] = 0
		configuration['mig_start'] = 0

	Trainer.set_seed(configuration['random_seed'])
	data = get_data_loader(configuration)
	PerfomanceLogger.configure_logger(configuration)
	perf_logger = PerfomanceLogger()
	if configuration['ablation']:
		run_ablation_wrapper(configuration, data, perf_logger)
	elif configuration['evaluation']:
		run_evaluation_wrapper(configuration, data, perf_logger)
	else:
		run_training_wrapper(configuration, data, perf_logger)


if __name__ == "__main__":
	config = get_config(sys.argv[1:])
	main(config)
