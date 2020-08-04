from absl.testing import absltest
import numpy as np
from mig import MIG
from data.dsprites import DSprites
from config import get_config
import sys
sys.path.insert(0,"../")


class MIGTest(absltest.TestCase):

	def test_metric(self):
		config = get_config(sys.argv[1:])
		data = DSprites(config)
		mig = MIG(data,0)
		representation_function = lambda x: x
		random_state = np.random.RandomState(0)
		scores = mig.compute_mig(representation_function, 3000, random_state)
		self.assertBetween(scores["discrete_mig"], 0.9, 1.0)

	def test_bad_metric(self):
		config = get_config()
		data = DSprites(config)
		mig = MIG(data)
		representation_function = np.zeros_like
		random_state = np.random.RandomState(0)
		scores = mig.compute_mig(representation_function,3000, random_state)
		self.assertBetween(scores["discrete_mig"], 0.0, 0.2)

	def test_duplicated_latent_space(self):
		config = get_config()
		data = DSprites(config)
		mig = MIG(data)
		def representation_function(x):
			x = np.array(x, dtype=np.float64)
			return np.hstack([x, x])
		random_state = np.random.RandomState(0)
		scores = MIG.compute_mig(representation_function,3000, random_state)
		self.assertBetween(scores["discrete_mig"], 0.0, 0.1)

if __name__ == "__main__":
	absltest.main()