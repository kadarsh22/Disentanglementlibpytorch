import torch.nn as nn



class InfonceEncoder(nn.Module):
	def __init__(self):
		super(InfonceEncoder, self).__init__()
		self.linear1_en = nn.Linear(2, 128)
		self.linear2_en = nn.Linear(128, 128)
		self.linear3_en = nn.Linear(128, 5)

	def forward(self,x):
		out = self.linear1_en(x)
		out = self.linear2_en(out)
		out = self.linear3_en(out)
		return out

