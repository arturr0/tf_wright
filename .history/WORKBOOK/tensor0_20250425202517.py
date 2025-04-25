import tensorflow as tf
import numpy as np
tensor = tf.zeros((1))
# tensor.eval()
print(tensor)
print(tensor.numpy())  # Get the NumPy array
constant = tf.constant(100)
print(constant.numpy())
rand = tf.random.normal((2, 2), mean=0.0, stddev=1.0)
print(rand)