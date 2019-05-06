import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
# 训练数据
train_file = open("C:/Users/maya/Desktop/mnist_train.csv","r")
train_list = train_file.readlines()
train_file.close()
# 测试数据
test_file = open("C:/Users/maya/Desktop/mnist_train.csv","r")
test_list = test_file.readlines()
test_file.close()

# x_train = np.asfarray(train_list,dtype=np.float).reshape()

model = Sequential([
    Dense(units=32,input_dim=784),
    Activation('relu'),
    Dense(units=10),
    Activation('softmax'),
])
# 优化器
rmsprop = RMSprop(lr=0.001,rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
print('--------Train----------')
model.fit()