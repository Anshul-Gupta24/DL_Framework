# performs stochastic gradient descent
def SGD(params, alpha=0.1):
	for layer_params in params:
		for p, grad in layer_params:
			p -= alpha * grad
