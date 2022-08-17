import torch.nn as nn
import torch

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
BATCH_SIZE = 128
# Hyper parameters
learning_rate = 0.001
num_epochs = 300

def train(model, x_test_cuda, y_test_cuda, inputs_cuda, labels_cuda):

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # loss function

    hidden_state = None

    history = []

    for epoch in range(num_epochs):
        model.train()
        output, _ = model(inputs_cuda, hidden_state)
        # print(output.size())

        # loss = criterion(output[:,0,:].view(-1), labels_cuda)
        loss = criterion(output.view(-1), labels_cuda)  # 计算预测值和label的差距，view(-1)表示把tensor进行flatten
        optimiser.zero_grad()
        loss.backward()  # back propagation
        optimiser.step()  # update the parameters

        if epoch % 20 == 0:
            model.eval()
            test_output, _ = model(x_test_cuda, hidden_state)
            test_loss = criterion(test_output.view(-1), y_test_cuda)
            print('epoch {}, loss {}, eval loss {}'.format(epoch, loss.item(), test_loss.item()))
        else:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
        history.append(loss.item())
