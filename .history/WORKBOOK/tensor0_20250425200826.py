import tensorflow as tf
import numpy as np
tensor = tf.zeros((3, 0, 3))
# tensor.eval()
print(tensor)
print(tensor.numpy())  # Get the NumPy array
