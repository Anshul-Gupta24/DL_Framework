from module import Module

class  MSE(Module):
	def  forward(self , input):
		pred, target = input
		self.pred = pred
		self.target = target
		loss = (pred - target).pow(2)

		return loss

	def  backward(self, gradwrtoutput):
		input_grad = 2*(self.pred - self.target)
		
		return gradwrtoutput * input_grad

	def  param(self):
		return  []