import torch
from torch import nn

class MultiheadSelfAttentionBlock(nn.Module):
	def __init__(
			self,
			embedding_dim: int = 768,
			num_heads: int = 12,
			attn_dropout: int = 0						
	):
		super().__init__()

		self.layer_norm = nn.LayerNorm(
			normalized_shape=embedding_dim
		)

		self.msa = nn.MultiheadAttention(
			embed_dim=embedding_dim,
			num_heads=num_heads,
			dropout=attn_dropout,
			batch_first=True
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x
