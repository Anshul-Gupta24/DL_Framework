import torch
from module import Module

class Sequential(Module):

	def __init__(self, layers):
		self.layers = layers

	def forward(self, input):
		for i, l in enumerate(self.layers):
			op = l.forward(input)
			input = op

		return op

	def backward(self, gradwrtoutput):
		layers_reverse = self.layers[::-1]
		for l in layers_reverse:
			gradwrtinput = l.backward(gradwrtoutput)
			gradwrtoutput = gradwrtinput

	def param(self):
		params = []
		for l in self.layers:
			params.append(l.param())

		return params

	def zero_grad(self):
		for l in self.layers:
			l.zero_grad()
