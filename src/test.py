import matplotlib.pyplot as py
import numpy as np
import tensorflow as tf
import base64
import cv2
import matplotlib.pyplot as plt
# a = np.array([[1],[2],[3],[4]])
# py.imshow(a,interpolation="nearest")
# py.show()
# a = np.random.rand(3,3)-0.5
# a = tf.zeros([10])
# ma1 = tf.constant([[3,3]])
# ma2 = tf.constant([[2],
#                    [2]])
# product = tf.matmul(ma1,ma2)
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)
# X = np.linspace(-1, 1, 20)
#
# np.random.shuffle(X)
# print(X)
#
# print(X[len(X)-5:])
# y_train = np.zeros(shape=(1,10))
# y_train[0,1] = 9
# print(y_train)
# img = cv2.imread('C:/Users/maya/Desktop/timg.jpg')
# cv2.imshow('src',img)
# print(img.shape)
# print(img.size)
# print(img.dtype)
# print(img)

image = plt.imread('C:/Users/maya/Desktop/timg.png',0)
plt.imshow(image,cmap="Greys",interpolation="None")
plt.show()