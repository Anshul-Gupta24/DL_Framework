import torch
from module import Module
import math

# Linear layer
class Linear(Module):

	def __init__(self, in_dim, out_dim):
		# initialize weights and biases		
		std = 1 / math.sqrt(in_dim)
		self.weights = torch.empty(out_dim, in_dim).uniform_(-std, std)
		self.biases = torch.empty(out_dim).uniform_(-std, std)
		
		self.weights_grad = torch.empty(self.weights.size()).zero_()
		self.biases_grad = torch.empty(self.biases.size()).zero_()

	def forward(self, input):
		self.input = input
		return self.weights.mv(input) + self.biases

	def backward(self, gradwrtoutput):
		self.weights_grad.add_(gradwrtoutput.view(-1, 1).mm(self.input.view(1, -1)))
		self.biases_grad.add_(gradwrtoutput)
		
		input_grad = self.weights.t().mv(gradwrtoutput)
		return input_grad

	def param(self):
		return [(self.weights, self.weights_grad), (self.biases, self.biases_grad)]

	def zero_grad(self):
		self.weights_grad = torch.empty(self.weights.size()).zero_()
		self.biases_grad = torch.empty(self.biases.size()).zero_()
