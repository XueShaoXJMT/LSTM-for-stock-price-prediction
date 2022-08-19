# LSTM-for-stock-price-prediction

This project used LSTM model to predicit stock price via time series prediction. As stock price is time series data, one day’s value is more related to its previous day’s value then the value 10 days ago, normal machine learning and deep learning frameworks cannot deal with it because they assume each piece of data has no relationship with others. To deal with the problem, I chose LSTM as it has memory to remember previous data.

# data split
We treat the problem as a regression problem, use a slicing window to predict one day’s value based on a sequence of value of its previous days. For example, if the window size is 5, we will predict each day’s value using its previous 5 days.

Based on this idea, we split the data into serval chunks, each chunk’s first 2/3 as training set, and the last 1/3 is validation set

# model
The LSTM model is based on Keras, it is a two hidden layer RNN of the following specifications:

layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))

layer 2 uses a fully connected module with one unit

the 'mean_squared_error' loss is used.

