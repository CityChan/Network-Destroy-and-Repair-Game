import numpy as np
import tensorflow as tf

m = np.arange(0, 30, 2)
m = m.reshape(3, 5)
m = np.matrix(m)
m = np.asarray(m)
print(m.shape)
m = tf.convert_to_tensor(m)
print(m)