import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# for modeling
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# read in the data
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv'
df = pd.read_csv(url)

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# split into X and Y
Y = df['species']
X = df.drop(['species'], axis=1)

print(X.shape)
print(Y.shape)

# convert to numpy arrays
X = np.array(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_Y = to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_Y)

# build a model
model = Sequential()
model.add(Dense(16,
                input_shape=(X.shape[1],),
                activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement
# in the validation loss for 10 consecutive epochs
es = EarlyStopping(mode='min',
                   patience=10,
                   restore_best_weights=True)

# now we just update the model fit call
history = model.fit(X,
                    dummy_Y,
                    callbacks=[es],
                    epochs=8000000,
                    batch_size=10,
                    validation_split=0.2,
                    verbose=1)

history_dict = history.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "r" is for "solid red line"
plt.plot(epochs, acc, 'r', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

preds = model.predict(X)
print(preds[0])
print(np.sum(preds[0]))

matrix = confusion_matrix(dummy_Y.argmax(axis=1),preds.argmax(axis=1))
print(classification_report(dummy_Y.argmax(axis=1), preds.argmax(axis=1)))