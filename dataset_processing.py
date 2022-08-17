from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess(dataset, sc):

    dataset_cl = dataset[dataset['Name'] == 'SWKS'].close.values
    # scale the data
    dataset_cl = dataset_cl.reshape(dataset_cl.shape[0], 1)
    dataset_cl = sc.fit_transform(dataset_cl)

    #Create a function to process the data into 7 day look back slices
    def processData(data, prev):
        X, Y = [], []
        for i in range(len(data) - prev - 1):
            X.append(data[i: (i + prev), 0])
            Y.append(data[(i + prev), 0])
        return np.array(X), np.array(Y)
    x, y = processData(dataset_cl, 7)
    return x, y
