import pandas as pd
from dataset_processing import *
from RNN import *
from train import *
from Test import *
from plot import *

# Importing the training set
dataset = data = pd.read_csv('./all_stocks_5yr.csv')
sc = MinMaxScaler(feature_range = (0, 1))
x, y = preprocess(dataset, sc)

# Split
x_train, x_test = x[:int(x.shape[0] * 0.80)], x[int(x.shape[0] * 0.80):]
y_train, y_test = y[:int(y.shape[0] * 0.80)], y[int(y.shape[0] * 0.80):]

# reshaping
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

x_test_cuda = torch.tensor(x_test).float().cuda()
y_test_cuda = torch.tensor(y_test).float().cuda()

# we use all the data in one batch
inputs_cuda = torch.tensor(x_train).float().cuda()
labels_cuda = torch.tensor(y_train).float().cuda()

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
BATCH_SIZE = 128
# Hyper parameters
learning_rate = 0.001
num_epochs = 300

rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, bidirectional=False)
rnn.cuda()

train(rnn, x_test_cuda, y_test_cuda, inputs_cuda, labels_cuda)
test(rnn, x_test_cuda, inputs_cuda)
plot_result(sc, rnn, x, y, x_test_cuda, inputs_cuda)












