







import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

# A. Data Preparation
sentences = [
    "I love machine learning",
    "I love deep learning",
    "machine learning is interesting",
    "deep learning is powerful",
    "I enjoy artificial intelligence",
    "machine learning is useful",
    "deep learning is amazing",
    "I study machine learning",
    "artificial intelligence is interesting",
    "I love artificial intelligence"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

vocab_size = len(tokenizer.word_index) + 1

# B. Generate Training Data
window_size = 1
X, y = [], []

for seq in sequences:
    for i in range(len(seq)):
        context = seq[max(0, i-window_size):i] + seq[i+1:i+window_size+1]

        if context:
            X.append(context)
            y.append(seq[i])

X = tf.keras.utils.pad_sequences(
    X, maxlen=2*window_size, padding='post'
)
y = np.array(y)

print("\nCBOW Training Pairs:")

for i in range(len(X)):
    context_words = [
        tokenizer.index_word.get(word, "PAD")
        for word in X[i] if word != 0
    ]

    target_word = tokenizer.index_word.get(y[i], "Unknown")

    print("Context:", context_words, " -> Target:", target_word)


# C. Build and Train CBOW Model
model = Sequential([
    Embedding(vocab_size, 16, mask_zero=True),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, y, epochs=300, verbose=1)

# D. Output / Prediction
context = ["I",  "learning"]
test = tokenizer.texts_to_sequences([" ".join(context)])
test = tf.keras.utils.pad_sequences(test, maxlen=2, padding='post')

prediction = model.predict(test, verbose=0)
word_id = np.argmax(prediction)

print("\nContext Words:", context)
print("Predicted Target Word:",
      tokenizer.index_word.get(word_id, "Unknown"))

# E. Model Evaluation
predictions = model.predict(X, verbose=0)
predicted_words = np.argmax(predictions, axis=1)

correct = np.sum(predicted_words == y)
incorrect = len(y) - correct

accuracy = correct / len(y) * 100

print("\nCorrect Predictions:", correct)
print("Incorrect Predictions:", incorrect)
print("Training Accuracy:", round(accuracy, 2), "%")

# F. Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
