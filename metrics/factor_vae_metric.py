import numpy as np
import logging
import torch



class FactorVAEMetric(object):
    """ Impementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """
    def __init__(self, dsprites, device_id):
        super(FactorVAEMetric, self).__init__()
        self.data = dsprites
        self.device_id = device_id


    def compute_factor_vae(self, model, random_state, batch_size ,num_train , num_eval , num_variance_estimate):
        # logging.info("Computing global variances to standardise.")
        global_variances = self._compute_variances(model, num_variance_estimate, random_state)
        active_dims = self._prune_dims(global_variances)
        if not active_dims.any():
            scores_dict = {"train_accuracy" : 0. ,"eval_accuracy" : 0. ,"num_active_dims" : 0}
            return scores_dict

        # logging.info("Generating training set.")

        training_votes = self._generate_training_batch(model, batch_size, num_train, random_state,
                                                  global_variances, active_dims)
        classifier = np.argmax(training_votes, axis=0)
        other_index = np.arange(training_votes.shape[1])

        # logging.info("Evaluate training set accuracy.")
        train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
        # logging.info("Training set accuracy: %.2g", train_accuracy)

        # logging.info("Generating evaluation set.")
        eval_votes = self._generate_training_batch(model, batch_size, num_eval, random_state,
                                              global_variances, active_dims)

        # logging.info("Evaluate evaluation set accuracy.")
        eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

        # logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
        scores_dict = {"train_accuracy": train_accuracy, "eval_accuracy": eval_accuracy, "num_active_dims": len(active_dims)}
        return scores_dict

    def _prune_dims(self,variances, threshold=0.0000001):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= threshold

    def _compute_variances(self, model, num_variance_estimate, random_state, eval_batch_size=64):

        latents = self.data.sample_latent(num_variance_estimate)
        observations = self.data.sample_images_from_latent(latents)
        representations , _ = model.encoder(torch.from_numpy(observations).cuda(self.device_id))
        assert representations.shape[0] == num_variance_estimate
        return np.var(representations.data.cpu().numpy(), axis=0, ddof=1)

    def _generate_training_sample(self, model,  batch_size, random_state, global_variances,
                                  active_dims):

        # Select random coordinate to keep fixed.
        factor_index = random_state.randint(low = 2, high = self.data.num_factors)
        # Sample two mini batches of latent variables.
        factors1 = self.data.sample_latent(batch_size)
        # Fix the selected factor across mini-batch.
        factors1[:, factor_index] = factors1[0, factor_index]
        observation = self.data.sample_images_from_latent(factors1)

        representations , _ = model.encoder(torch.from_numpy(observation))
        representations = representations.data.cpu().numpy()
        ## Rescaling
        local_variances = np.var(representations, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims])
        return factor_index, argmin     ## armin+2 incase we are using true latent since we are not considering first 2 latents(background and shape)

    def _generate_training_batch(self, model,
                                 batch_size, num_points, random_state,
                                 global_variances, active_dims):

        votes = np.zeros((self.data.num_factors, global_variances.shape[0]), dtype=np.int64)
        for _ in range(num_points):
            factor_index, argmin = self._generate_training_sample(model, batch_size, random_state,
                                                             global_variances, active_dims)
            votes[factor_index, argmin] += 1
        return votes