import torch
import argparse
torch.manual_seed(10)
torch.set_grad_enabled(False)

from linear import Linear
from activation import ReLU, Tanh
from sequential import Sequential
from sgd import SGD
from loss import LossMSE
from generate_data import get_data


def train(model, args, X_train, Y_train, X_test, Y_test):

	num_samples = X_train.shape[0]
	num_batches = num_samples // args.batch_size

	# define loss
	mse = LossMSE()
	for ep in range(args.epochs):
		idx = 0
		train_loss = 0
		for nb in range(num_batches):
			for i in range(args.batch_size):
				# forward pass
				op = model.forward(X_train[idx])
				loss = mse.forward((op, Y_train[idx]))
				train_loss += loss
				
				# backward pass				
				model.backward(mse.backward())
				
				idx += 1

			# update model params for each batch
			SGD(model.param(), alpha=0.001)
			model.zero_grad()
		
		train_loss = train_loss / (num_batches * args.batch_size)
		print('Avg train loss at epoch {}: {:.2f}'.format(ep, train_loss.item()))
		test_error = test(model, X_test, Y_test)
		print('Test error at epoch {}: {:.2f}%'.format(ep, test_error.item()*100))
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

	parser = argparse.ArgumentParser(description='Mini Project 2 arguments.')
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--num_runs', type = int, default = 1, help = 'Number of runs of a model')
	args = parser.parse_args()
	
	train_errors = torch.empty(args.num_runs); test_errors = torch.empty(args.num_runs)
	for i in range(args.num_runs):
		print('Run: ', i)

		# generate training and test data
		X_train, Y_train = get_data(n=1000)
		X_test, Y_test = get_data(n=1000)

		# define model
		model = Sequential([
				Linear(2, 25),
				ReLU(),
				Linear(25, 25),
				ReLU(),
				Linear(25, 25),
				ReLU(),
				Linear(25, 1),
				Tanh()
				])

		# train model
		train(model, args, X_train, Y_train, X_test, Y_test)
	
		# compute final train and test errors	
		final_train_error = test(model, X_train, Y_train)
		final_test_error = test(model, X_test, Y_test)
		print('Final train error is: {:.2f}'.format(final_train_error*100))
		print('Final test error is: {:.2f}'.format(final_test_error))
		print()
		train_errors[i] = final_train_error
		test_errors[i] = final_test_error

	if args.num_runs > 1:
		print('Train error mean: {:.2f}'.format(train_errors.mean()*100))
		print('Train error std dev: {:.2f}'.format(train_errors.std()*100))
		print('Test error mean: {:.2f}'.format(test_errors.mean()*100))
		print('Test error std dev: {:.2f}'.format(test_errors.std()*100))
