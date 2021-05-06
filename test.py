import torch

from linear import Linear
from activation import ReLU, Tanh
from sequential import Sequential
from sgd import SGD
from generate_data import get_data


def train(model, X_train, Y_train, X_test, Y_test):

	num_epochs = 100
	batch_size = 16

	num_samples = X_train.shape[0]
	num_batches = num_samples // batch_size

	for ep in range(num_epochs):
		idx = 0
		train_loss = 0
		for nb in range(num_batches):
			for i in range(batch_size):
				op = model.forward(X_train[idx])

				loss = (op - Y_train[idx]).pow(2) 
				train_loss += loss
				model.backward(2*(op - Y_train[idx]))
				
				idx += 1

			SGD(model.param(), alpha=0.001)
			model.zero_grad()
		
		train_loss = train_loss / (num_batches * batch_size)
		print('Avg loss at epoch ', ep, ': ', train_loss)
		test_error = test(model, X_test, Y_test)
		print('Test error at epoch ', ep, ': ', test_error)
		print()


def test(model, X, Y):

	num_samples = X.shape[0]
	
	test_error = 0
	for idx in range(num_samples):
		op = model.forward(X[idx])

		p = (op > 0.5).float()
		test_error += (p!=Y[idx])

	return test_error / num_samples


if __name__=='__main__':

	X_train, Y_train = get_data(n=1000)
	X_test, Y_test = get_data(n=1000)

	model = Sequential([
			Linear(2, 25),
			ReLU(),
			Linear(25, 25),
			ReLU(),
			Linear(25, 25),
			ReLU(),
			Linear(25, 1),
			ReLU()
			])
	
	train(model, X_train, Y_train, X_test, Y_test)
	final_train_error = test(model, X_train, Y_train)
	final_test_error = test(model, X_test, Y_test)
	print('Final train error is: ', final_train_error)
	print('Final test error is: ', final_test_error)
