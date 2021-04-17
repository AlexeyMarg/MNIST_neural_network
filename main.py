import numpy as np
import matplotlib.pyplot as plt
from neural_network import neuralNetwork

# dataset files location
train_data_location = 'dataset/mnist_train.csv'
test_data_location = 'dataset/mnist_test.csv'

def load_train_data(train_data_location):
    train_data_file = open(train_data_location, 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    return train_data_list

def load_test_data(test_data_location):
    test_data_file = open(test_data_location, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    return test_data_list

def show_image(data):
    values = data.split(',')
    image_array = np.asfarray(values[1:]).reshape(28, 28)
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

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

# load test data
test_data_list = load_test_data(test_data_location)

correct_number = 0
incorrect_number = 0
confusion_matrix = np.zeros((10, 10))

print('Start testing')
for sample in test_data_list:
    values = sample.split(',')
    correct_label = int(values[0])
    inputs = np.asfarray(values[1:]) / 255.0 * 0.99 + 0.01
    outputs = n.run(inputs)
    label = np.argmax(outputs)

    if label == correct_label:
        correct_number += 1
    else:
        incorrect_number += 1
    confusion_matrix[int(correct_label)][int(label)] += 1

print('Number of correct answers: ', correct_number)
print('Number of incorrect answers: ', incorrect_number)
print('Accuracy: ', correct_number/(correct_number+incorrect_number))

for i in range(10):
    precision_denominator = 0
    recall_denominator = 0
    for j in range(10):
        precision_denominator += confusion_matrix[i][j]
        recall_denominator += confusion_matrix[j][i]
    print('Statistics for ', i,' detection' )
    print('Presision: ', confusion_matrix[i][i]/precision_denominator)
    print('Recall: ', confusion_matrix[i][i]/recall_denominator)
