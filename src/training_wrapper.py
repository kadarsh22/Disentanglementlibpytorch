from model_loader import get_model
from train import Trainer
from evalutation import Evaluator
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging
import time
import numpy as np


class log_gaussian:

	def __call__(self, x, mu):
		var = torch.ones(mu.shape[0], mu.shape[1]).cuda()
		logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
		        (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

		return logli.sum(1).mean().mul(-1)


def run_training_wrapper(configuration, data, perf_logger):
	for key, values in configuration.items():
		logging.info(' {} : {}'.format(key, values))
	save_config(configuration)
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	model, d_optimizer , g_optimizer = get_model(configuration)
	model_trainer = Trainer(data, configuration)
	evaluator = Evaluator(data, configuration)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")

	images = data.images
	train_loader = torch.utils.data.DataLoader(images, batch_size=configuration['batch_size'], shuffle=True)
	device = torch.device('cuda:' + str(configuration['device_id']))


	for i in range(configuration['epochs']):
		if configuration['model_arch'] == 'vae':
			model.train()
			model, loss, optimizer = model_trainer.train_vae(model, optimizer[0], i)
		else:
			model.encoder.train()
			model.decoder.train()
			# model, loss, optimizer = model_trainer.train_gan(model, optimizer, i)

			start_time = time.time()
			d_loss, g_loss, info_loss = 0, 0, 0
			model.encoder.to(device)
			model.decoder.to(device)

			adversarial_loss = torch.nn.BCELoss()
			criterionQ_con = log_gaussian()

			for iter, images in enumerate(train_loader):
				images = images.type(torch.FloatTensor).to(device)
				z_noise = torch.rand(configuration['batch_size'], configuration['noise_dim'],dtype=torch.float32, device=device)* 2 - 1
				c_cond = torch.rand(configuration['batch_size'], configuration['latent_dim'],dtype=torch.float32, device=device) * 2 - 1

				z = torch.cat((z_noise, c_cond), dim=1)

				g_optimizer.zero_grad()

				fake_x = model.decoder(z)
				latent_code , prob_fake = model.encoder(fake_x)
				label_real = torch.full((configuration['batch_size'],), 1, dtype=torch.float32, device=device)
				g_loss = adversarial_loss(prob_fake,label_real)
				cont_loss = criterionQ_con(c_cond,latent_code)

				G_loss = g_loss + cont_loss*0.05
				G_loss.backward()

				g_optimizer.step()

				d_optimizer.zero_grad()
				latent_code , prob_real = model.encoder(images)
				label_real = torch.full((configuration['batch_size'],), 1, dtype=torch.float32, device=device)
				loss_real = adversarial_loss(prob_real, label_real)
				loss_real.backward()


				fake_x = model.decoder(z)
				latent_code_gen , prob_fake = model.encoder(fake_x, spectral_norm =False)
				label_fake = torch.full((configuration['batch_size'],), 0, dtype=torch.float32, device=device)
				loss_fake = adversarial_loss(prob_fake, label_fake)
				q_loss = criterionQ_con(c_cond,latent_code_gen)
				loss = loss_fake + 0.05*q_loss
				loss.backward()

				D_loss = loss_real + loss
				d_optimizer.step()


				d_loss = d_loss + D_loss.item()
				g_loss = g_loss + G_loss.item()
			#
			logging.info("Epochs  %d / %d Time taken %d sec  D_Loss: %.3f, G_Loss %.3F" % (
				i, configuration['epochs'], time.time() - start_time,
				g_loss / len(train_loader), d_loss / len(train_loader)))


		if i % configuration['saving_freq'] == 0 and i != 0:
			perf_logger.start_monitoring("Saving Model")
			# saver.save_model(model, optimizer, loss, epoch=i)
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
