from model_loader import get_model
from evalutation import Evaluator
from train import Trainer
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging
import matplotlib.pyplot as plt
import torchvision



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

    model, optimizer, loss = saver.load_model(model=model, optimizer=optimizer)
    latents_sampled = data.sample_latent(size=100)
    latents_sampled[:, 1] = 1
    indices_sampled = data.latent_to_index(latents_sampled)
    imgs_sampled = data.images[indices_sampled]
    model.encoder.cuda()
    representations , _ = model.encoder(torch.from_numpy(imgs_sampled))
    grid_img = torchvision.utils.make_grid(torch.Tensor(imgs_sampled).view(-1, 1, 64, 64), nrow=10, padding=5, pad_value=1)
    grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
    plt.imsave('image1.png', grid.numpy())
    metrics = evaluator.evaluate_model(model, epoch=0)
    z, _ = model.encoder(torch.from_numpy(data.images[0]).type(torch.FloatTensor))
    visualise_results.visualise_latent_traversal(z, model.decoder, 0)
    saver.save_results(metrics, 'metrics')
