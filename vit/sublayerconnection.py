from torch import nn

from .layernorm import LayerNorm 

class SublayerConnection(nn.Module):
	def __init__(self, size: int, dropout: float):
		super(SublayerConnection, self).__init__()

		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))