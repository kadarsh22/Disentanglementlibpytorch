from absl.testing import absltest
import numpy as np
from factor_vae_metric import FactorVAEMetric
from data.dsprites import DSprites
from config import get_config
import sys
sys.path.insert(0,"../")


class FactorVaeTest(absltest.TestCase):
  def test_metric(self):
    config = get_config()
    data = DSprites(config)
    factor_vae = FactorVAEMetric(data)
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = factor_vae.compute_factor_vae(representation_function, random_state, 128, 2000, 2000,2000)
    self.assertBetween(scores["train_accuracy"], 0.9, 1.0)
    self.assertBetween(scores["eval_accuracy"], 0.9, 1.0)


if __name__ == "__main__":
  absltest.main()