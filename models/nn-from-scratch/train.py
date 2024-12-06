"""
train.py

This script trains a simple 3-layer neural network on the MNIST dataset. It uses
a small subset of the dataset (100 samples) for demonstration purposes. After
training, the model is saved in multiple formats for later use.

Classes:
- NeuralNetwork: Imported from neural_network.py for training.

Dependencies:
- numpy
- pickle
- json
- csv
- neural_network.py

Inputs:
- data/mnist_train_100.csv: A CSV file containing training data.

Outputs:
- outputs/models/trained_network.pkl: A file containing the trained neural network model (Pickle).
- outputs/models/model_metadata.json: A JSON file with model metadata and parameters.
- outputs/csv/weights.csv: A CSV file with the neural network's weights.
- outputs/csv/biases.csv: A CSV file with the neural network's biases.
"""

import numpy as np
import pickle  # For saving the trained model in binary format
import json    # For saving model metadata
import csv     # For saving weights and biases
from neural_network import NeuralNetwork

# Define key parameters for the neural network
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10
LEARNING_RATE = 0.1
EPOCHS = 5

# File paths
TRAINING_DATA_FILE = "data/mnist_train_100.csv"
MODEL_PICKLE_FILE = "outputs/models/trained_network.pkl"
MODEL_JSON_FILE = "outputs/models/model_metadata.json"
WEIGHTS_CSV_FILE = "outputs/csv/weights.csv"
BIASES_CSV_FILE = "outputs/csv/biases.csv"


def load_training_data(file_path):
    """
    Loads training data from a CSV file.
    """
    with open(file_path, "r") as file:
        return file.readlines()


def preprocess_data(record, output_nodes):
    """
    Preprocesses a single record from the dataset.
    """
    all_values = record.split(',')
    inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    return inputs, targets


def train_network(network, training_data, epochs):
    """
    Trains the neural network on the provided dataset.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for record in training_data:
            inputs, targets = preprocess_data(record, OUTPUT_NODES)
            network.train(inputs, targets)


def save_model_to_pickle(network, file_path):
    """
    Saves the trained neural network to a file using pickle.
    """
    with open(file_path, "wb") as model_file:  # "wb" is write-binary mode
        pickle.dump(network, model_file)
    print(f"Model saved to {file_path} (Pickle)")


def save_model_metadata_to_json(network, file_path):
    """
    Saves metadata (e.g., dimensions, learning rate, weights, biases) of the neural network to a JSON file.

    Parameters:
    - network: Trained neural network instance.
    - file_path (str): Path to save the JSON metadata.
    """
    metadata = {
        "input_nodes": network.input_nodes,
        "hidden_nodes": network.hidden_nodes,
        "output_nodes": network.output_nodes,
        "learning_rate": network.lr,
        "layer_dimensions": {
            "input_to_hidden": {
                "weights_shape": network.wih.shape,  # Shape of weight matrix
                "biases_count": network.hidden_nodes  # Bias count matches hidden layer neurons
            },
            "hidden_to_output": {
                "weights_shape": network.who.shape,  # Shape of weight matrix
                "biases_count": network.output_nodes  # Bias count matches output layer neurons
            }
        },
        "weights_input_hidden": network.wih.tolist(),
        "weights_hidden_output": network.who.tolist(),
    }
    with open(file_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Model metadata saved to {file_path} (JSON)")


def save_weights_and_biases_to_csv(network, weights_file, biases_file):
    """
    Saves the neural network's weights and biases to CSV files.
    """
    # Save weights
    with open(weights_file, "w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Input-to-Hidden Weights"])
        writer.writerows(network.wih)
        writer.writerow(["Hidden-to-Output Weights"])
        writer.writerows(network.who)
    print(f"Weights saved to {weights_file} (CSV)")

    # Save biases (if applicable)
    # For simplicity, we save placeholders as "biases.csv" since the current network does not explicitly track biases.
    with open(biases_file, "w", newline="") as bf:
        writer = csv.writer(bf)
        writer.writerow(["Biases Placeholder"])
        writer.writerow(["Biases are implicit in the weights update formulas."])
    print(f"Biases saved to {biases_file} (CSV)")


def main():
    """
    Main function to train the neural network and save the model in multiple formats.
    """
    # Create an instance of the neural network
    network = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # Load and preprocess training data
    training_data_list = load_training_data(TRAINING_DATA_FILE)

    # Train the neural network
    train_network(network, training_data_list, EPOCHS)

    # Save the trained model in multiple formats
    save_model_to_pickle(network, MODEL_PICKLE_FILE)
    save_model_metadata_to_json(network, MODEL_JSON_FILE)
    save_weights_and_biases_to_csv(network, WEIGHTS_CSV_FILE, BIASES_CSV_FILE)


if __name__ == "__main__":
    main()