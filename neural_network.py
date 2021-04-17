import numpy as np
from scipy.special import expit


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # number of input layer nodes
        self.inodes = input_nodes
        # number of hidden layer nodes
        self.hnodes = hidden_nodes
        # number of output layer nodes
        self.onodes = output_nodes
        # learning rate
        self.l_rate = learning_rate
        # input to hidden layer weights in (-0.5; 0.5)
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # hidden to output layer weights in (-0.5; 0.5)
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        # sigmoid activation function
        self.activation_function = lambda x: expit(x)
    # training with one sample function
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # forward propogation
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        output_inputs = np.dot(self.who, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)

        # errors calculation
        output_errors = targets - output_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # weights correction
        self.who += self.l_rate * np.dot((output_errors * output_outputs * (1.0 - output_outputs)), np.transpose(hidden_outputs))
        self.wih += self.l_rater * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    # run trained neural network function
    def run(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        output_inputs = np.dot(self.who, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)

        return output_outputs
