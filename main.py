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


n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
