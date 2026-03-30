import torch
from torch import nn

from .attention import attention
from .clone import clones

class MultiHeadAttention(nn.Module):
	def __init__(self, h: int, d_model: int, dropout: float = 0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
		if mask is not None:
			mask = mask.unsqueeze(1)

		nbatches = query.size(0)

		query, key, value = [
			lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			for lin, x in zip(self.linears, (query, key, value))
		]

		x, self.attn = attention(
			query, key, value, mask=mask, dropout=self.dropout
		)

		x = (
			x.transpose(1, 2)
				.contiguous()
				.view(nbatches, -1, self.h * self.d_k)
		)

		del query
		del key 
		del value
		return self.linears[-1](x)