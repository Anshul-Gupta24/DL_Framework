import torch
from module import Module

# non-linear activation functions
class ReLU(Module):

	def __init__(self):
		return

	def forward(self, input):
		self.input = input
		return input * (input > 0)

	def backward(self, gradwrtoutput):
		grad = (self.input > 0).float()
		
		input_grad = gradwrtoutput * grad
		return input_grad

	def param(self):
		return []

	def zero_grad(self):
		return


class Tanh(Module):

	def __init__(self):
		return

	def forward(self, input):
		self.input = input
		return input.tanh()

	def backward(self, gradwrtoutput):
		grad = 4 * (self.input.exp() + self.input.mul(-1).exp()).pow(-2)
		
		input_grad = gradwrtoutput * grad
		return input_grad

	def param(self):
		return []

	def zero_grad(self):
		return
