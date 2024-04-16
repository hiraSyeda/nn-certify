import sys
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading MNIST dataset")

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


print("Building the training pipeline")
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

print("Building the evaluation pipeline")
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

print("Creating the model")
inputs = Input((28, 28))
z = Flatten()(inputs)
z = Dense(64, activation='relu')(z)
outputs = Dense(10)(z)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print("Training the model")
model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test,
    verbose='false',
)

history = model.history.history

# learning curve
# accuracy
acc = history['sparse_categorical_accuracy']
val_acc = history['val_sparse_categorical_accuracy']

# loss
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)
plt.subplot(2, 1, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

model.summary()
model.save("model.keras")
with open('model.summary', 'w') as sys.stdout:
    model.summary()

# Extract weights and biases
weights_and_biases = [
    (layer.get_weights()[0], layer.get_weights()[1])
    for layer in model.layers if len(layer.get_weights()) > 0]

for i, (weights, biases) in enumerate(weights_and_biases):
    np.savez(f"layer_{i}_weights_biases.npz", weights=weights, biases=biases)

# Create a directory to save the files if it does not exist
save_dir = 'model_weights_csv'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through each layer, extract weights and biases, and save them
for i, (weights, biases) in enumerate(weights_and_biases):
    np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%f')
    np.savetxt(os.path.join(save_dir, f'layer_{i}_biases.csv'), biases, delimiter=',', fmt='%f')

# ...reset 'sys.stdout' for later script output
sys.stdout = sys.__stdout__

print("Evaluating the testing dataset")

model.evaluate(ds_test)
print("Calculating Confusion Matrix")

labels = []
predicted_classes = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ResourceWarning)

for image_batch, label_batch in ds_test:
    predictions = model.predict(image_batch)
    predicted_labels = np.argmax(predictions, axis=1)
    labels.extend(label_batch.numpy())
    predicted_classes.extend(predicted_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels, predicted_classes)
print("Confusion Matrix")
print(conf_matrix)

try:
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.show()
except KeyboardInterrupt:
    print("Plotting was interrupted.")
finally:
    plt.close('all')

# At the end of your script
K.clear_session()
