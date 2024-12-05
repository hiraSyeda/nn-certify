# Neural Networks: Basics and Components

## 1. What is a Neural Network (NN)?
A **neural network (NN)** is a computational model inspired by the way biological neural networks process information. It is used for tasks such as classification, regression, feature extraction, and more.

### Key Characteristics:
- Consists of layers of neurons (nodes) connected via weights.
- Learns patterns from data using forward and backward propagation.

---

## 2. Typical Components of a Neural Network

### 2.1 Layers
1. **Input Layer**:
   - Takes the input features (e.g., pixels in an image, tabular data columns).
   - No learnable parameters in this layer.

2. **Hidden Layers**:
   - One or more layers where computations occur.
   - Neurons apply weights, biases, and activation functions to transform input data.

3. **Output Layer**:
   - Produces predictions or results (e.g., probabilities, class labels).
   - Activation function depends on the task:
     - Classification: `softmax`, `sigmoid`
     - Regression: None or `linear`

---

### 2.2 Parameters in a Neural Network
1. **Weights (`w`)**:
   - Learnable parameters that determine the strength of connections between neurons.
   - Initialized randomly and updated during training.

2. **Biases (`b`)**:
   - Learnable parameters added to weighted inputs for flexibility.
   - Each neuron has a unique bias.

3. **Activation Functions**:
   - Non-linear transformations applied to neuron outputs.
   - Examples:
     - **ReLU**: `max(0, x)` (for hidden layers)
     - **Sigmoid**: `1 / (1 + exp(-x))` (for binary classification)
     - **Softmax**: For multi-class classification.

---

## 2.3 Parameters by Layer

Here’s a breakdown of the parameters and components in each type of layer:

| **Layer**         | **Parameters/Components**                                                  | **Description**                                                                                          | **Examples of Activation Function**    |
|-------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------|
| **Input Layer**   | None                                                                       | - This layer only passes the input features to the next layer.                                           | None (just passes raw data)            |
| **Hidden Layers** | - **Weights (`w`)** <br> - **Biases (`b`)** <br> - **Activation Function** | - Learnable parameters (`w` and `b`) to transform inputs. <br> - Activation introduces non-linearity.    | **ReLU**, `tanh`, `sigmoid`            |
| **Output Layer**  | - **Weights (`w`)** <br> - **Biases (`b`)** <br> - **Activation Function** | - Similar to hidden layers, but the activation depends on the task (e.g., classification or regression). | **Softmax** (multi-class), **Sigmoid** |

## 3. Representation of a Trained Neural Network

After training, a neural network stores its **learned parameters** (weights and biases) for each layer. These parameters define how the network transforms input data into predictions. They can be saved in a file (e.g., CSV) for visualization, debugging, or reuse in future tasks.

---

### 3.1 What Keras Generates

In Keras, the learned parameters can be accessed using the `get_weights()` method. This method provides:

1. **Weights**: Represent the connections between neurons in adjacent layers. These are stored as 2D NumPy arrays with dimensions `(input_neurons, output_neurons)`.
2. **Biases**: Represent learnable offsets applied to each neuron. These are stored as 1D NumPy arrays with dimensions `(output_neurons)`.

When calling `model.get_weights()`, Keras outputs a **list** where:
- Odd-indexed arrays represent **weights** for each layer.
- Even-indexed arrays represent **biases** for each layer.

#### Example: Output of `get_weights()`

Let’s consider a simple neural network:
- **Input Layer**: 2 inputs (no weights or biases).
- **Hidden Layer 1**: 3 neurons, connected to the input layer.
- **Hidden Layer 2**: 2 neurons, connected to Hidden Layer 1.
- **Output Layer**: 1 neuron, connected to Hidden Layer 2.

After training, calling `model.get_weights()` might produce:

```python
[
    array([[ 0.25, -0.30,  0.50], [ 0.60, -0.05,  0.40]]),  # Weights for Hidden Layer 1
    array([ 0.10, -0.05,  0.20]),                           # Biases for Hidden Layer 1
    array([[ 0.40, -0.50], [ 0.70,  0.80], [ 0.60, -0.10]]),# Weights for Hidden Layer 2
    array([ 0.30,  0.10]),                                 # Biases for Hidden Layer 2
    array([[ 0.90], [-0.20]]),                             # Weights for Output Layer
    array([ 0.15])                                         # Biases for Output Layer
]
```
#### Explanation of Each Array

#### 1. Hidden Layer 1:
- **Weights**: `(2, 3)` (2 inputs, 3 neurons).
- **Biases**: `(3, )` (1 bias per neuron).

#### 2. Hidden Layer 2:
- **Weights**: `(3, 2)` (3 inputs, 2 neurons).
- **Biases**: `(2, )`.

#### 3. Output Layer:
- **Weights**: `(2, 1)` (2 inputs, 1 neuron).
- **Biases**: `(1, )`.

---

## 4. Typical Workflow of a Neural Network in Python
The workflow generally includes:
1. **Data Preparation**:
   - Preprocessing (e.g., normalization, encoding labels).
   - Splitting into training and testing sets.

2. **Model Design**:
   - Architecture with input, hidden, and output layers.
   - Specify number of neurons and activation functions.

3. **Model Compilation**:
   - Define:
     - **Loss function**: e.g., `binary_crossentropy`, `mse`
     - **Optimizer**: e.g., `adam`, `sgd`
     - **Metrics**: e.g., `accuracy`

4. **Model Training**:
   - `epochs`: Number of training iterations.
   - `batch_size`: Number of samples per training step.

5. **Model Evaluation and Prediction**:
   - Evaluate the trained model using test data.
   - Make predictions.

---