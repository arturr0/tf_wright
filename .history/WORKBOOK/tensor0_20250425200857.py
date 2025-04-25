import tensorflow as tf
import numpy as np
tensor = tf.zeros((3, 3, 0))
# tensor.eval()
print(tensor)
print(tensor.numpy())  # Get the NumPy array
