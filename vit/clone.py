import copy
from torch import nn

def clones(module: nn.Module, N: int):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])