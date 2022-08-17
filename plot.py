import matplotlib as plt
import numpy  as np

def plot_result(sc, model, x, y, x_test_cuda, inputs_cuda):
	# plot original data
    plt.plot(sc.inverse_transform(y.reshape(-1,1)), color='k')

    train_pred, hidden_state = model(inputs_cuda, None)
    train_pred_cpu = train_pred.cpu().detach().numpy()

    # use hidden state from previous training data
    test_predict, _ = model(x_test_cuda, hidden_state)
    test_predict_cpu = test_predict.cpu().detach().numpy()

    # plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
    split_pt = int(x.shape[0] * 0.80) + 7 # window_size
    plt.plot(np.arange(7, split_pt, 1), sc.inverse_transform(train_pred_cpu.reshape(-1,1)), color='b')
    plt.plot(np.arange(split_pt, split_pt + len(test_predict_cpu), 1), sc.inverse_transform(test_predict_cpu.reshape(-1,1)), color='r')

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel('price of MMM stock')
    plt.legend(['original series','training fit','testing fit'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()