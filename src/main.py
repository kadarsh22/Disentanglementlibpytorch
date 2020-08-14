import sys
from config import get_config
from data_loader import get_data_loader
from train import Trainer
from training_wrapper import run_training_wrapper
from ablation_wrapper import run_ablation_wrapper
from evaluation_wrapper import run_evaluation_wrapper
from logger import PerfomanceLogger


def main(configuration):
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
