import tensorflow as tf
import numpy as np
tensor = tf.zeros((1))
# tensor.eval()
# print(tensor)
# print(tensor.numpy())  # Get the NumPy array
# constant = tf.constant(100)
# print(constant.numpy())
# rand_normal = tf.random.normal((2, 2), mean=0.0, stddev=1.0)
# rand_uniform = tf.random.uniform((2, 2), minval=0.0, maxval=1.0)
# print(rand_uniform)
t = tf.range(1, 5, 1)
td = tf.linalg.diag(t)
tu = tf.eye(4)
# print(tu)
ttu = tf.linalg.matrix_transpose(td)
print(ttu)