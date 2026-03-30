import torch
from torch import nn

class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout = 0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
		self.layers = nn.Sequential(
			self.w_1,
			nn.ReLU(),
			self.dropout,
			self.w_2,
		)

	def forward(self, x: torch.Tensor):
		return self.layers(x)