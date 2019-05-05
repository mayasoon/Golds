import matplotlib.pyplot as py
import numpy as np
import tensorflow as tf

# a = np.array([[1],[2],[3],[4]])
# py.imshow(a,interpolation="nearest")
# py.show()
# a = np.random.rand(3,3)-0.5
# a = tf.zeros([10])
ma1 = tf.constant([[3,3]])
ma2 = tf.constant([[2],
                   [2]])
product = tf.matmul(ma1,ma2)
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
