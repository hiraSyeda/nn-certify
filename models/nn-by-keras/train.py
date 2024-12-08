import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Define key parameters for the neural network
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10
EPOCHS = 10
BATCH_SIZE = 32

# Step 1: Load and Preprocess the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten 28x28 images into 784-dimensional vectors and normalize
x_train = x_train.reshape(-1, INPUT_NODES).astype('float32') / 255.0
x_test = x_test.reshape(-1, INPUT_NODES).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, OUTPUT_NODES)
y_test = tf.keras.utils.to_categorical(y_test, OUTPUT_NODES)

# Step 2: Define the Model
model = Sequential([
    Dense(HIDDEN_NODES, activation='relu', input_shape=(INPUT_NODES,)),  # Input to Hidden Layer
    Dense(OUTPUT_NODES, activation='softmax')                           # Hidden to Output Layer
])

# Step 3: Compile the Model
model.compile(
    optimizer='adam',                           # Adaptive optimizer
    loss='categorical_crossentropy',            # Loss function for classification
    metrics=['accuracy']                        # Metric to monitor
)

# Step 4: Train the Model
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 6: Save the Model
model.save("mnist_nn_model.h5")
print("Model saved to mnist_nn_model.h5")