import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hat_function(x):
	return max(0, 1 - abs(x))
    
class HatApproxNet(nn.Module):
	def __init__(self, hidden_dim = 8):
		super(HatApproxNet, self).__init__()
		self.D1 = nn.Linear(1, hidden_dim)
		self.D2 = nn.Linear(hidden_dim, hidden_dim)
		self.D3 = nn.Linear(hidden_dim, hidden_dim)
		self.D4 = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		x = torch.tanh(self.D1(x))
		x = torch.tanh(self.D2(x))
		x = torch.tanh(self.D3(x))
		x = torch.tanh(self.D4(x))
		return x
