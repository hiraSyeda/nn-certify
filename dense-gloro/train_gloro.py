import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from gloro.utils import print_if_verbose
from gloro.utils import get_data
from gloro.utils import get_optimizer
from gloro.models import GloroNet
from gloro.layers import Dense
from gloro.layers import Flatten
from gloro.layers import Input
from gloro.training import losses
from tensorflow.keras import backend as K
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import LrScheduler
from gloro.training.callbacks import TradesScheduler
from gloro.training.metrics import rejection_rate
from sklearn.metrics import confusion_matrix


def train_gloro(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=None,
        batch_size=None,
        optimizer='adam',
        lr=0.001,
        lr_schedule='fixed',
        trades_schedule=None,
        verbose=True,
):
    _print = print_if_verbose(verbose)

    # Load data and set up data pipeline.
    _print('loading data...')

    train, test, metadata = get_data(dataset, batch_size, augmentation)

    # Create the model.
    _print('creating model...')

    inputs = Input((28, 28))
    z = Flatten()(inputs)
    z = Dense(64, activation='relu')(z)
    outputs = Dense(10)(z)

    g = GloroNet(inputs, outputs, epsilon)

    if verbose:
        g.summary()

    # Compile and train the model.
    _print('compiling model...')

    g.compile(
        # loss=losses.get(loss),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=get_optimizer(optimizer, lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        # metrics=[rejection_rate]
    )
    g.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=[
                      EpsilonScheduler(epsilon_schedule),
                      LrScheduler(lr_schedule),
                  ] + ([TradesScheduler(trades_schedule)] if trades_schedule else []),
    )

    return g


def script(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=100,
        batch_size=128,
        optimizer='adam',
        lr=1e-3,
        lr_schedule='decay_to_0.000001',
        trades_schedule=None,
):

    g = train_gloro(
        dataset,
        epsilon,
        epsilon_schedule=epsilon_schedule,
        loss=loss,
        augmentation=augmentation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        lr_schedule=lr_schedule,
        trades_schedule=trades_schedule,
    )

    history = g.history.history

    # learning curve
    # accuracy
    acc = history['sparse_categorical_accuracy']
    val_acc = history['val_sparse_categorical_accuracy']

    # loss
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(epochs, acc, 'r', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    ax2.plot(epochs, loss, 'r', label='Training Loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    plt.tight_layout()
    plt.show()
    fig.savefig('learning_curve.png')

    g.summary()
    g.save("model.keras")
    with open('gloro.summary', 'w') as sys.stdout:
        g.summary()

    # Extract weights and biases
    weights_and_biases = [
        (layer.get_weights()[0], layer.get_weights()[1])
        for layer in g.layers if len(layer.get_weights()) > 0]

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
    train, ds_test, metadata = get_data(dataset, batch_size, augmentation)

    g.evaluate(ds_test)

    print("Calculating Confusion Matrix")

    labels = []
    predicted_classes = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)

    for image_batch, label_batch in ds_test:
        predictions = g.predict(image_batch)
        predicted_labels = np.argmax(predictions, axis=1)
        labels.extend(label_batch.numpy())
        predicted_classes.extend(predicted_labels)

    print(np.max(labels))
    print(np.max(predicted_classes))

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
        plt.xticks(np.arange(11))
        plt.yticks(np.arange(10))
        plt.savefig('confusion_matrix.png')
        plt.show()
    except KeyboardInterrupt:
        print("Plotting was interrupted.")
    finally:
        plt.close('all')

    # At the end of your script
    K.clear_session()

