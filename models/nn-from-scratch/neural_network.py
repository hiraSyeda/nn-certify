"""
neural_network.py

This module defines a simple 3-layer neural network implementation for
training on the MNIST dataset. It includes methods for training the network
and querying it to make predictions.

Classes:
- NeuralNetwork: A class implementing the neural network.

Dependencies:
- numpy
- scipy.special (for the sigmoid activation function)
"""

import numpy as np
from scipy.special import expit  # Sigmoid function for activation

"""
neural_network.py

This module defines a simple 3-layer neural network implementation for 
training on the MNIST dataset. It includes methods for training the network
and querying it to make predictions.

Classes:
- NeuralNetwork: A class implementing the neural network.

Dependencies:
- numpy
- scipy.special (for the sigmoid activation function)
"""

import numpy as np
from scipy.special import expit  # Sigmoid function for activation


class NeuralNetwork:
    """A simple 3-layer neural network."""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        Initialize the neural network.

        Parameters:
        - input_nodes: Number of input nodes (features).
        - hidden_nodes: Number of hidden layer nodes.
        - output_nodes: Number of output nodes (classes).
        - learning_rate: Learning rate for weight updates.
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weight matrices with random values (normal distribution)
        # Scale weights by 1/sqrt(number of incoming connections) to avoid vanishing/exploding gradients
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5),
                                    (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                    (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate  # Learning rate

        # Activation function: Sigmoid
        self.activation_function = expit

    def train(self, inputs_list, targets_list):
        """
        Train the neural network using backpropagation.

        Parameters:
        - inputs_list: List of input values.
        - targets_list: List of target output values.
        """
        # Convert input and target lists to 2D arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Forward pass: Calculate signals for hidden and output layers
        hidden_inputs = np.dot(self.wih, inputs)  # Signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # Signals out of hidden layer

        final_inputs = np.dot(self.who, hidden_outputs)  # Signals into final layer
        final_outputs = self.activation_function(final_inputs)  # Signals out of final layer

        # Compute output layer error (target - actual)
        output_errors = targets - final_outputs
        # Compute hidden layer error (backpropagated from output layer)
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update weights: Hidden-to-output
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),  # Error gradient
            hidden_outputs.T
        )

        # Update weights: Input-to-hidden
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),  # Error gradient
            inputs.T
        )

    def query(self, inputs_list):
        """
        Query the neural network (feed-forward).

        Parameters:
        - inputs_list: List of input values.

        Returns:
        - Outputs from the final layer.
        """
        # Convert input list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T

        # Forward pass: Calculate signals for hidden and output layers
        hidden_inputs = np.dot(self.wih, inputs)  # Signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # Signals out of hidden layer

        final_inputs = np.dot(self.who, hidden_outputs)  # Signals into final layer
        final_outputs = self.activation_function(final_inputs)  # Signals out of final layer

        return final_outputs