import torch
import math

# samples points uniformly from a unit square
# points within the defined circle have label 1; points outside have label 0
def get_data(n):

	X = torch.empty(n, 2).uniform_(0, 1)
	
	center = torch.empty(2).fill_(0.5)
	radius_2 = torch.empty(1).fill_(1 / (2*math.pi))

	in_circle = (X - center).pow(2).sum(1) - radius_2
	in_circle = in_circle < 0
	Y = in_circle.float()

	return X, Y	


