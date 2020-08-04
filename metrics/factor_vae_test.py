from absl.testing import absltest
import numpy as np
from factor_vae_metric import FactorVAEMetric
from data.dsprites import DSprites
from config import get_config
import sys
sys.path.insert(0,"../")


class FactorVaeTest(absltest.TestCase):
  def test_metric(self):
    config = get_config(sys.argv[1:])
    data = DSprites(config)
    factor_vae = FactorVAEMetric(data,0)
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = factor_vae.compute_factor_vae(representation_function, random_state, 128, 2000, 2000,2000)
    print(scores)
    self.assertBetween(scores["train_error_rate"], 0, 0.2)
    self.assertBetween(scores["eval_error_rate"], 0, 0.2)


if __name__ == "__main__":
  absltest.main()