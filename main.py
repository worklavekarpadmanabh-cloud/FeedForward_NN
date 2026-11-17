import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# b. Load training and testing data
train_data= pd.read_csv("E:\FeedForward_NN\MNIST\mnist_train.csv")
test_data= pd.read_csv("E:\FeedForward_NN\MNIST\mnist_test.csv")

# Separate labels and images
X_train= train_data.iloc[:,1:].values
y_train= train_data.iloc[:, 0].values

X_test= test_data.iloc[:, 1:].values
y_test= test_data.iloc[:,0].values

# Normalize pixel values
X_train= X_train/255
X_test= X_test/255

# c. Define Feed-Forward Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer= tf.keras.optimizers.SGD(learning_rate=0.01),
    loss= 'sparse_categorical_crossentropy',
    metrics= ['accuracy']
)

history= model.fit(
    X_train,y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test,y_test)
)

loss, accuracy= model.evaluate(X_test,y_test)

print("Test loss", loss)
print("Test accuracy", accuracy*100,"%")

# f. Plot training loss
plt.plot(history.history['loss'],label='Tranning loss')
plt.plot(history.history['val_loss'], label='validation loss')

plt.title("Tranning and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot Tranning accuracy
plt.plot(history.history['accuracy'], label='Tranning accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Tranning and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




# Predict one test image

prediction = model.predict(X_test[:1])

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])






















"""
# a. Import necessary packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# b. Load training and testing data

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Normalize pixel values

X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images (28x28 into 784)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)


# c. Define Feed-Forward Neural Network

model= Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128,activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()


# d. Train the model using SGD

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)"""