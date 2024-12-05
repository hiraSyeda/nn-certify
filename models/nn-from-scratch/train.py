"""
train.py

This script trains a simple 3-layer neural network on the MNIST dataset. It uses
a small subset of the dataset (100 samples) for demonstration purposes.

Classes:
- NeuralNetwork: Imported from neural_network.py for training.

Dependencies:
- numpy
- neural_network.py

Inputs:
- mnist_train_100.csv: A CSV file containing training data.

Outputs:
- Trained weights and biases (stored in the NeuralNetwork instance).
"""

import numpy as np
from neural_network import NeuralNetwork

# Define key parameters for the neural network
INPUT_NODES = 784  # Number of input features (28x28 pixel images in MNIST)
HIDDEN_NODES = 200  # Number of neurons in the hidden layer
OUTPUT_NODES = 10  # Number of output classes (digits 0–9)
LEARNING_RATE = 0.1  # Learning rate for the neural network

# Number of epochs (iterations over the dataset)
EPOCHS = 5

# File containing training data
TRAINING_DATA_FILE = "mnist_train_100.csv"

def load_training_data(file_path):
    """
    Loads training data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - list: A list of strings, each representing a row of the CSV file.
    """
    with open(file_path, "r") as file:
        return file.readlines()

def preprocess_data(record, output_nodes):
    """
    Preprocesses a single record from the dataset.

    Parameters:
    - record (str): A single row of the CSV file.
    - output_nodes (int): Number of output nodes in the neural network.

    Returns:
    - tuple: (inputs, targets)
      - inputs (ndarray): Normalized input values (0.01 to 1.0).
      - targets (ndarray): Target output values (0.01 for all except the desired label, which is 0.99).
    """
    # Split the record into individual values
    all_values = record.split(',')

    # Normalize input values (scale to 0.01–1.0)
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # Create target outputs (all 0.01, except the correct label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99

    return inputs, targets

def train_network(network, training_data, epochs):
    """
    Trains the neural network on the provided dataset.

    Parameters:
    - network (NeuralNetwork): An instance of the NeuralNetwork class.
    - training_data (list): List of training data records.
    - epochs (int): Number of times to iterate over the dataset.
    """
    print("Training the neural network...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for record in training_data:
            # Preprocess the record to get inputs and targets
            inputs, targets = preprocess_data(record, OUTPUT_NODES)
            # Train the network with the inputs and targets
            network.train(inputs, targets)
    print("Training complete!")

def main():
    """
    Main function to train the neural network.
    """
    # Create an instance of the neural network
    network = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # Load and preprocess training data
    training_data_list = load_training_data(TRAINING_DATA_FILE)

    # Train the neural network
    train_network(network, training_data_list, EPOCHS)

if __name__ == "__main__":
    main()