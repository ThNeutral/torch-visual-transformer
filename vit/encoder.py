from .helpers.clone import clones
from torch import nn

class Encoder(nn.Module):
	def __init__(self, layer: nn.Module, N: int):
		super(Encoder, self).__init__()

		self.layers = clones(layer, N)
		self.norm = nn.LayerNorm(layer.size)

	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)