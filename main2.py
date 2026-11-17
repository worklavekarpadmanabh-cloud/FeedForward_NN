import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

train_data= pd.read_csv("E:\FeedForward_NN\MNIST\mnist_train.csv")
test_data= pd.read_csv("E:\FeedForward_NN\MNIST\mnist_test.csv")

X_train= train_data.iloc[:,1:].values
y_train= train_data.iloc[:, 0].values

X_test= test_data.iloc[:, 1:].values
y_test= test_data.iloc[:,0].values

X_train= X_train/255
X_test= X_test/255

model= Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation= 'softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history= model.fit(
    X_train,y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

loss,accuracy= model.evaluate(X_test,y_test)

print("Tets loss", loss)
print("test accyracy", accuracy)

plt.plot(history.history['loss'], label="tranning loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title("Modal loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label="tranning accuracy")
plt.plot(history.history['val_accuracy'], label="validation accuarcy")
plt.title("Model acuuracy")
plt.xlabel("epchos")
plt.ylabel("accuracy")
plt.legend()
plt.show()































"""
# a. Import necessary packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Stage 1: Loading and preprocessing image data

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Normalize pixel values

X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images (28x28 into 784)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)


# Stage 2: Defining the model architecture

model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()


# Stage 3: Training the model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


# Stage 4: Estimating model performance

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")


# Additional evaluation: Confusion Matrix and Classification Report

predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report:")
print(classification_report(y_test, predictions))


# Plot training and validation loss

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Plot training and validation accuracy

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Additional: Predict one test image

prediction = model.predict(X_test[:1])

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])


# Display the test image

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}, Actual: {y_test[0]}")
plt.axis('off')
plt.show()"""

