


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Load and preprocess dataset

data = pd.read_csv(r"E:\FeedForward_NN\ecg.csv", header=None).values

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y
)

# Normalize data
minimum = X_train.min()
maximum = X_train.max()

X_train = (X_train - minimum) / (maximum - minimum)
X_test = (X_test - minimum) / (maximum - minimum)

# Select normal ECG signals 

X_normal = X_train[y_train == 1]

print("Normal training data:", X_normal.shape)
print("Testing data:", X_test.shape)
print("Actual Normal:", np.sum(y_test == 1))
print("Actual Anomalies:", np.sum(y_test == 0))




encoder = Sequential([
    Input(shape=(140,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu')
])

decoder = Sequential([
    Input(shape=(16,)),
    Dense(32, activation='relu'),
    Dense(140, activation='sigmoid')
])

autoencoder = Sequential([encoder, decoder])


#  D. Compile the model
#Why did we use Adam?
#1. It automatically adjusts the learning rate for individual parameters.
#2. It generally provides fast and effective training.

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


#  E. Train the model

autoencoder.fit(
    X_normal, X_normal,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)


#  F. Detect anomalies

reconstructed = autoencoder.predict(X_test, verbose=0)
errors = np.mean(np.abs(X_test - reconstructed), axis=1)

normal_reconstructed = autoencoder.predict(X_normal, verbose=0)
normal_errors = np.mean(
    np.abs(X_normal - normal_reconstructed), axis=1
)

threshold = np.mean(normal_errors) + np.std(normal_errors)

predictions = (errors > threshold).astype(int)
actual = (y_test == 0).astype(int)

print("Anomaly Threshold:", threshold)
print("Predicted Normal:", np.sum(predictions == 0))
print("Predicted Anomalies:", np.sum(predictions == 1))
print("Correct Predictions:", np.sum(predictions == actual))
print("Incorrect Predictions:", np.sum(predictions != actual))
print("Accuracy:", np.mean(predictions == actual) * 100, "%")
print("Actual Normal:", np.sum(actual == 0))
print("Actual Anomalies:", np.sum(actual == 1))

# Evaluation Metrics

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predictions))

print("\nClassification Report:")
print(classification_report(
    actual, predictions,
    target_names=["Normal", "Anomaly"]
))



























"""
# A. Import necessary packages

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# B. Load dataset using URL

url = "PASTE_DIRECT_CSV_URL_HERE"

data = pd.read_csv(url, header=None).values

print("Dataset Shape:", data.shape)


# Separate features and labels

X = data[:, :-1]
y = data[:, -1]


# Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=21,
    stratify=y
)


# Normalize data

minimum = X_train.min()
maximum = X_train.max()

X_train = (X_train - minimum) / (maximum - minimum)
X_test = (X_test - minimum) / (maximum - minimum)


# Select normal ECG signals for training

X_normal = X_train[y_train == 1]

print("Normal training data:", X_normal.shape)
print("Testing data:", X_test.shape)

print("Actual Normal:", np.sum(y_test == 1))
print("Actual Anomalies:", np.sum(y_test == 0))


# C. Define Autoencoder Architecture

# Encoder

encoder = Sequential([
    Input(shape=(140,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu')
])

# Decoder

decoder = Sequential([
    Input(shape=(16,)),
    Dense(32, activation='relu'),
    Dense(140, activation='sigmoid')
])

# Combine Encoder and Decoder

autoencoder = Sequential([encoder, decoder])

autoencoder.summary()


# D. Compile the model

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# E. Train the Autoencoder using normal ECG data

history = autoencoder.fit(
    X_normal, X_normal,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)


# F. Detect anomalies

# Reconstruct test ECG signals

reconstructed = autoencoder.predict(X_test, verbose=0)

# Calculate reconstruction error

errors = np.mean(
    np.abs(X_test - reconstructed),
    axis=1
)


# Calculate reconstruction error for normal training data

normal_reconstructed = autoencoder.predict(
    X_normal, verbose=0
)

normal_errors = np.mean(
    np.abs(X_normal - normal_reconstructed),
    axis=1
)


# Calculate anomaly threshold

threshold = np.mean(normal_errors) + np.std(normal_errors)


# Predict anomalies

predictions = (errors > threshold).astype(int)

# Convert actual labels:
# 1 = Normal, 0 = Anomaly
# For evaluation:
# 0 = Normal, 1 = Anomaly

actual = (y_test == 0).astype(int)


# Display results

print("Anomaly Threshold:", threshold)

print("Predicted Normal:", np.sum(predictions == 0))
print("Predicted Anomalies:", np.sum(predictions == 1))

print("Correct Predictions:", np.sum(predictions == actual))
print("Incorrect Predictions:", np.sum(predictions != actual))

print("Accuracy:", np.mean(predictions == actual) * 100, "%")

print("Actual Normal:", np.sum(actual == 0))
print("Actual Anomalies:", np.sum(actual == 1))


# G. Evaluation Metrics

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predictions))

print("\nClassification Report:")
print(classification_report(
    actual,
    predictions,
    target_names=["Normal", "Anomaly"]
))"""













"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input




train = pd.read_csv(r"E:\FeedForward_NN\MNIST\mnist_train.csv")
test = pd.read_csv(r"E:\FeedForward_NN\MNIST\mnist_test.csv")

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

# Use digit 0 as normal data
X_normal = X_train[y_train == 0]




encoder = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(32, activation='relu')
])

decoder = Sequential([
    Input(shape=(32,)),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')
])

autoencoder = Sequential([encoder, decoder])



autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)




autoencoder.fit(
    X_normal, X_normal,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)




reconstructed = autoencoder.predict(X_test, verbose=0)
errors = np.mean((X_test - reconstructed) ** 2, axis=1)

normal_reconstructed = autoencoder.predict(X_normal, verbose=0)
normal_errors = np.mean((X_normal - normal_reconstructed) ** 2, axis=1)

threshold = np.percentile(normal_errors, 95)

predictions = (errors > threshold).astype(int)
actual = (y_test != 0).astype(int)

print("Anomaly Threshold:", threshold)
print("Total Normal Images:", np.sum(predictions == 0))
print("Total Anomalies:", np.sum(predictions == 1))
print("Actual Normal Images:", np.sum(actual == 0))
print("Actual Anomalies:", np.sum(actual == 1))
print("Correct Predictions:", np.sum(predictions == actual))
print("Incorrect Predictions:", np.sum(predictions != actual))
print("Accuracy:", np.mean(predictions == actual) * 100, "%")"""


