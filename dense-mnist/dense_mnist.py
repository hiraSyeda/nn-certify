import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

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
plt.ylabel('Accuracy')

plt.show()

model.summary()
model.save("model.keras")
with open('model.summary', 'w') as sys.stdout:
    model.summary()

# ...reset 'sys.stdout' for later script output
sys.stdout = sys.__stdout__

print("Evaluating the testing dataset")

result = model.evaluate(ds_test)
print("Confusion Matrix")
predictions = model.predict(ds_test)
predicted_labels = np.argmax(predictions, axis=1)

ds = ds_test.take(1)
test_labels = tfds.as_numpy(ds)

'''
# Compute the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for MNIST Classification')
plt.show()
'''

