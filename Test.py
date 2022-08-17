import torch
import matplotlib.pyplot as plt

HIDDEN_SIZE = 64

def mean_squared_error(actual, predicted):
	sum_square_error = 0.0
	for i in range(len(actual)):
		sum_square_error += (actual[i] - predicted[i])**2.0
	mean_square_error = 1.0 / len(actual) * sum_square_error
	return torch.mean(mean_square_error)

from math import log
# calculate binary cross entropy
def binary_cross_entropy(actual, predicted):
	sum_score = 0.0
	for i in range(len(actual)):
		sum_score += actual[i] * log(1e-15 + predicted[i])
	mean_sum_score = 1.0 / len(actual) * sum_score
	return -mean_sum_score

# calculate categorical cross entropy
def categorical_cross_entropy(actual, predicted):
	sum_score = 0.0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
			sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
	mean_sum_score = 1.0 / len(actual) * sum_score
	return -mean_sum_score

def test(model, x_test_cuda, inputs_cuda):
	model.eval()

	train_pred, hidden_state = model(inputs_cuda, None)
	train_pred_cpu = train_pred.cpu().detach().numpy()

	# use hidden state from previous training data
	test_predict, _ = model(x_test_cuda, hidden_state)
	test_predict_cpu = test_predict.cpu().detach().numpy()

	m = mean_squared_error(train_pred, inputs_cuda)
	n = mean_squared_error(test_predict, x_test_cuda)
	print("training rmse: ")
	print(m)
	print("testing rmse: ")
	print(n)