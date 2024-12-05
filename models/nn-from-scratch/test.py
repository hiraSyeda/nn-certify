"""
test.py

This script tests the performance of a trained neural network on the MNIST dataset
using a small subset (10 samples). The network is imported from the `train` module.

Classes:
- NeuralNetwork: Imported from train.py.

Dependencies:
- numpy
- train.py

Inputs:
- mnist_test_10.csv: A CSV file containing test data.

Outputs:
- Performance score: Fraction of correct predictions.
"""

import numpy as np
from train import network  # Import the trained neural network instance

# File containing test data
TEST_DATA_FILE = "mnist_test_10.csv"


def load_test_data(file_path):
    """
    Loads test data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - list: A list of strings, each representing a row of the CSV file.
    """
    with open(file_path, "r") as file:
        return file.readlines()


def preprocess_data(record):
    """
    Preprocesses a single record from the test dataset.

    Parameters:
    - record (str): A single row of the CSV file.

    Returns:
    - tuple: (inputs, correct_label)
      - inputs (ndarray): Normalized input values (0.01 to 1.0).
      - correct_label (int): The actual label (0–9) from the dataset.
    """
    # Split the record into individual values
    all_values = record.split(',')
    correct_label = int(all_values[0])  # First value is the correct label
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # Normalize inputs
    return inputs, correct_label


def test_network(network, test_data):
    """
    Tests the neural network on the provided dataset and calculates its performance.

    Parameters:
    - network (NeuralNetwork): An instance of the NeuralNetwork class.
    - test_data (list): List of test data records.

    Returns:
    - float: The performance score (fraction of correct predictions).
    """
    scorecard = []  # Initialize scorecard

    for record in test_data:
        # Preprocess the record to get inputs and correct label
        inputs, correct_label = preprocess_data(record)

        # Query the network to get predictions
        outputs = network.query(inputs)
        predicted_label = np.argmax(outputs)  # Predicted label (highest value index)

        # Check if the prediction matches the correct label
        if predicted_label == correct_label:
            scorecard.append(1)  # Correct prediction
        else:
            scorecard.append(0)  # Incorrect prediction

    # Calculate performance as the fraction of correct predictions
    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    return performance


def main():
    """
    Main function to test the neural network and print its performance.
    """
    # Load test data
    test_data_list = load_test_data(TEST_DATA_FILE)

    # Test the neural network
    performance = test_network(network, test_data_list)
    print(f"Performance: {performance:.2%}")


if __name__ == "__main__":
    main()