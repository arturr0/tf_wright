import tensorflow as tf
import numpy as np

# Input data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Define a simple perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid')  # 1 neuron
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=500, verbose=0)

# Test the model
print("\nPredictions:")
predictions = model.predict(X)
for i, prediction in enumerate(predictions):
    print(f"Input: {X[i]} => Prediction: {prediction[0]:.4f}")
