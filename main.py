import numpy as np
from neural_network import neuralNetwork

# dataset files location
train_data_location = 'dataset/mnist_train.csv'
test_data_location = 'dataset/mnist_test.csv'

def load_train_data(train_data_location):
    train_data_file = open(train_data_location, 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    return train_data_list



# number of input nodes is a number of pixels
input_nodes = 784
hidden_nodes = 100
# each output node corresponds to particular number
output_nodes = 10
learning_rate = 0.3
# number of training epochs
epochs = 2

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load train data
train_data_list = load_train_data(train_data_location)
# start training
print('Start training')
for i in range(epochs):
    for sample in train_data_list:
        #split sample
        values = sample.split(',')
        # scaling of inputs to the range (0.01, 1)
        inputs = np.asfarray(values[1:])/255.0 * 0.99 +0.01
        # choose 0.01 target value for outputs that doesn't correspond to label
        targets = np.zeros(output_nodes) + 0.01
        # and 0.99 otherwise
        targets[int(values[0])] = 0.99
        # start train with scaled data
        n.train(inputs, targets)
print('End of training')
