import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
# 训练数据
train_file = open("C:/Users/admin/Desktop/mnist_train_100.csv","r")
train_list = train_file.readlines()
train_file.close()
train_data = np.asfarray(train_list[0].split(",")[1:]).reshape(1,784)/255
# 第一个Y值
y = train_list[0]
y_train = np.zeros(shape=(1,10))
y_train[0,int(y[0])] = 1
# y_train = np.asarray(y_train)
pass
for i in range(len(train_list)):
    if i > 0:
        arr = np.asfarray(train_list[i].split(',')[1:]).reshape(1,784)/255
        train_data = np.append(train_data,arr,axis=0)
        target = np.zeros(shape=(1, 10))
        s = train_list[i]
        target[0,int(s[0])] = 1
        y_train = np.append(y_train,target,axis=0)

# print('y_train = ',y_train)
# 测试数据
test_file = open("C:/Users/admin/Desktop/mnist_test_10.csv","r")
test_list = test_file.readlines()
test_file.close()
test_data = np.asfarray(train_list[0].split(",")[1:]).reshape(1,784)/255

t = train_list[0]
y_test = np.zeros(shape=(1,10))
y_test[0,int(t[0])] = 1

for i in range(len(test_list)):
    if i > 0:
        arr = np.asfarray(test_list[i].split(',')[1:]).reshape(1,784)/255
        test_data = np.append(test_data,arr,axis=0)
        target = np.zeros(shape=(1, 10))
        s = train_list[i]
        target[0, int(s[0])] = 1
        y_test = np.append(y_test, target, axis=0)

# print('y_test = ',y_test)
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
model.fit(train_data,y_train,epochs=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(test_data, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)