import torch
from torch import nn

class EmbeddingLayer(nn.Module):
	"""Turns a 2D input image into a 1D sequence learnable embedding vector.

  Args:
      c (int): Number of color channels for the input images. Defaults to 3.
      patch_size (int): Size of patches to convert input image into. Defaults to 16.
      d (int): Size of embedding to turn image into. Defaults to 768.
  """
	def __init__(self, c: int = 3, d: int = 768, patch_size: int = 16):
		super().__init__()

		self.patcher = nn.Conv2d(
			in_channels=c,
			out_channels=d,
			kernel_size=patch_size,
			stride=patch_size,
			padding=0
		)

		self.flatten = nn.Flatten(
			start_dim=2,
			end_dim=3
		)

		self.patch_size = patch_size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		image_resolution = x.shape[-1]
		assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

		
		return self.flatten(self.patcher(x)).permute(0, 2, 1)