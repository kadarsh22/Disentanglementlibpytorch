from visualiser import Visualiser
from training_wrapper import run_wrapper


def run_ablation_wrapper(configuration, data, perf_logger):
	count = 0
	keys = []
	list_vals = []
	for key, values in configuration.items():
		if isinstance(values, list):
			count = count + 1
			keys.append(key)
			list_vals.append(values)
	if count == 1:
		old_experiment_name = configuration['experiment_name']
		visualise_ablation_results = Visualiser(configuration)
		bce_loss, kld_loss, loss_total, beta_vae, factor_vae, mig = [], [], [], [], [], []
		for i in list_vals[0]:
			configuration['experiment_name'] = old_experiment_name + '/' + old_experiment_name + '_' + str(
				keys[0][:-1]) + '_' + str(i)
			configuration[str(keys[0][:-1])] = i
			loss, metrics = run_wrapper(configuration, data, perf_logger)
			if configuration['model_arch'] == 'vae':
				loss_total.append(loss['loss'])
				bce_loss.append(loss['bce_loss'])
				kld_loss.append(loss['kld_loss'])
			elif configuration['model_arch'] == 'gan':
				raise NotImplementedError
			else:
				raise NotImplementedError
			beta_vae.append(metrics['beta_vae'])
			factor_vae.append(metrics['factor_vae'])
			mig.append(metrics['mig'])

		x = [i for i in range(configuration['epochs'])]
		x_axis = [x] * len(beta_vae)
		visualise_ablation_results.visualise_ablation_results(loss_total, x_axis, 'total_loss', list_vals[0])
		visualise_ablation_results.visualise_ablation_results(bce_loss, x_axis, 'bce_loss', list_vals[0])
		visualise_ablation_results.visualise_ablation_results(kld_loss, x_axis, 'kld_loss', list_vals[0])
		visualise_ablation_results.visualise_ablation_results(beta_vae, x_axis, 'beta_vae', list_vals[0])
		visualise_ablation_results.visualise_ablation_results(factor_vae, x_axis, 'factor_vae', list_vals[0])
		visualise_ablation_results.visualise_ablation_results(mig, x_axis, 'mig', list_vals[0])
