"""
test.py

This script tests the performance of a saved neural network model on the MNIST dataset.
It uses a small subset of the dataset (10 samples) for demonstration purposes. The
trained model is loaded from a file, and its performance is evaluated. Results are
saved in JSON and CSV formats.

Functions:
- load_model: Loads the trained model from a file.
- preprocess_data: Prepares input data for querying the network.
- test_network: Tests the network and calculates its performance.
- save_test_results: Saves test results to JSON and CSV files.

Dependencies:
- numpy
- pickle
- csv
- json

Inputs:
- data/mnist_test_10.csv: A CSV file containing test data.
- outputs/models/trained_network.pkl: A file containing the trained neural network model.

Outputs:
- outputs/models/test_results.json: A file containing test results (performance and predictions).
- outputs/csv/test_predictions.csv: A CSV file containing detailed predictions for each sample.
"""

import numpy as np
import pickle
import json
import csv

# File paths
TEST_DATA_FILE = "data/mnist_test_10.csv"
MODEL_FILE = "outputs/models/trained_network.pkl"
TEST_RESULTS_JSON = "outputs/models/test_results.json"
TEST_PREDICTIONS_CSV = "outputs/csv/test_predictions.csv"


def load_test_data(file_path):
    """
    Loads test data from a CSV file.
    """
    with open(file_path, "r") as file:
        return file.readlines()


def preprocess_data(record):
    """
    Preprocesses a single record from the test dataset.
    """
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    return inputs, correct_label


def load_model(file_path):
    """
    Loads the trained neural network model from a file.
    """
    with open(file_path, "rb") as model_file:  # "rb" is read-binary mode
        return pickle.load(model_file)


def test_network(network, test_data):
    """
    Tests the neural network on the provided dataset and calculates its performance.

    Returns:
    - performance (float): Fraction of correct predictions.
    - predictions (list): A list of dictionaries containing details for each sample.
    """
    scorecard = []
    predictions = []

    for record in test_data:
        inputs, correct_label = preprocess_data(record)
        outputs = network.query(inputs)
        predicted_label = np.argmax(outputs)
        scorecard.append(1 if predicted_label == correct_label else 0)

        predictions.append({
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "outputs": outputs.tolist()
        })

    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    return performance, predictions

def save_test_results(performance, predictions, json_file, csv_file):
    """
    Saves test results to a JSON file and a CSV file.
    """
    # Convert NumPy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, (np.int64, np.float64)):  # Handle NumPy scalar types
            return obj.item()
        return obj

    # Save performance and predictions to JSON
    results = {
        "performance": float(performance),  # Ensure performance is a float
        "predictions": [
            {k: convert_to_serializable(v) for k, v in prediction.items()}
            for prediction in predictions
        ],
    }
    with open(json_file, "w") as jf:
        json.dump(results, jf, indent=4)
    print(f"Test results saved to {json_file} (JSON)")

    # Save predictions to CSV
    with open(csv_file, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Correct Label", "Predicted Label", "Outputs"])
        for prediction in predictions:
            writer.writerow([
                prediction["correct_label"],
                prediction["predicted_label"],
                prediction["outputs"]
            ])
    print(f"Test predictions saved to {csv_file} (CSV)")

def main():
    """
    Main function to load the trained model, test it, and save results.
    """
    # Load the trained neural network model
    network = load_model(MODEL_FILE)
    print(f"Model loaded from {MODEL_FILE}")

    # Load test data
    test_data_list = load_test_data(TEST_DATA_FILE)

    # Test the neural network
    performance, predictions = test_network(network, test_data_list)
    print(f"Performance: {performance:.2%}")

    # Save the test results
    save_test_results(performance, predictions, TEST_RESULTS_JSON, TEST_PREDICTIONS_CSV)


if __name__ == "__main__":
    main()