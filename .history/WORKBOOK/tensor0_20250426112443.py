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
# t_reshaped = tf.reshape(t, (4, 1))  # now shape (4,1)
# t2 = tf.constant([[1, 7], [3, 2]])
# t3 = tf.constant([[1, 7], [3, 2]])
# print(t3)
# rows = 3
# cols = 4
# my_array = []
# for i in range(rows):
#     row = []
#     for j in range(cols):
#         value = i * 10 + j  # or whatever formula you want
#         row.append(value)
#     my_array.append(row)
# t2t = tf.linalg.matrix_transpose(my_array)
# print(t2t)
# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([5, 6])
# c = tf.multiply(a, b)
# print(c)

a = tf.ones(2)
print(a)
b = tf.expand_dims(a, 1)
print(b)
c = tf.squeeze(b)
print(c) 

sum = b + c
print(sum)