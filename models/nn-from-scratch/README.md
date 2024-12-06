
# README for Neural Network from Scratch

---

## Project: NN-From-Scratch

This project demonstrates the development of a 3-layer neural network built from scratch to classify handwritten digits from the MNIST dataset. It includes custom implementations of neural network training, testing, and model evaluation.

---

## Features and Highlights

### 1. 3-Layer Neural Network:
- **Input Layer:** **784 neurons** (one for each pixel in a 28x28 image).
- **Hidden Layer:** **200 neurons** (to learn intermediate features).
- **Output Layer:** **10 neurons** (one for each digit 0–9).

### 2. Model Dimensions:
- **Input-to-Hidden Layer:**
  - **Weights Shape:** (200, 784).
  - **Total Weights:** 156,800.
  - **Weight Range:** ~[-0.11, 0.14].
  - **Biases Count:** 200.
- **Hidden-to-Output Layer:**
  - **Weights Shape:** (10, 200).
  - **Total Weights:** 2,000.
  - **Weight Range:** ~[-0.2, 0.16].
  - **Biases Count:** 10.

### 3. Performance:
- Achieved an accuracy of **60%** on a small subset of the MNIST test dataset (10 samples).

### 4. Training Highlights:
- The model was trained on a limited dataset of 100 samples for demonstration purposes.
- **Learning Rate:** 0.1.
- **Epochs:** 5 iterations over the training dataset.

---

## Project Structure

```
nn-from-scratch/
├── data/
│   ├── mnist_train_100.csv    # Training dataset
│   ├── mnist_test_10.csv      # Test dataset
├── outputs/
│   ├── csv/
│   │   ├── weights.csv        # Exported weights
│   │   ├── biases.csv         # Exported biases
│   │   ├── test_predictions.csv  # Test predictions
│   ├── models/
│       ├── trained_network.pkl  # Pickle of the trained model
│       ├── model_metadata.json  # Metadata of the model
│       ├── test_results.json    # Test results
├── neural_network.py           # Neural network implementation
├── train.py                    # Script to train the model
├── test.py                     # Script to test the model
```

---

## How to Run

### 0. Setup the Virtual Environment
The virtual environment (`venv`) for this project is located in the parent folder outside `nn-from-scratch` (e.g., `neural-networks/venv`).

1. Activate the virtual environment:
   ```bash
   source ../venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Navigate back to the `nn-from-scratch` directory:
   ```bash
   cd nn-from-scratch
   ```

### 1. Training the Model
To train the model:
```bash
python train.py
```
- The trained model will be saved to `outputs/models/trained_network.pkl`.
- Metadata and weights/biases are also saved for interpretability.

### 2. Testing the Model
To test the model:
```bash
python test.py
```
- Performance and predictions will be saved to `outputs/models/test_results.json`.

---

## Insights and Takeaways

- **Total Number of Weights:** 158,800 (156,800 for Input-to-Hidden, 2,000 for Hidden-to-Output).
- **Total Number of Biases:** 210 (200 for Hidden, 10 for Output).
- **Weight Ranges:**
  - Input-to-Hidden Layer: ~[-0.11, 0.14].
  - Hidden-to-Output Layer: ~[-0.2, 0.16].
- **Biases:** Represented as scalars and match the number of neurons in the respective layers.

### Key Learnings:
- Even a simple 3-layer network can effectively classify MNIST digits with reasonable accuracy given limited data.
- Exported weights, biases, and metadata provide transparency and reproducibility of the model's structure.
- This project is a foundational step in understanding neural networks without relying on external libraries like TensorFlow or PyTorch.