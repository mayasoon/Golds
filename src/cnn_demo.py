import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras import losses

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1,28, 28)
X_test = X_test.reshape(-1, 1,28, 28)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# print('X_train shape ',X_train.shape)
# print('y_train shape ',y_train.shape)
# print('X_test shape ',X_test.shape)
# print('y_test shape ',y_test.shape)
#
# print(y_test[0])
model = Sequential()
# 第一次滤波 得到32层
model.add(Convolution2D(
    filters=32,
    kernel_size=5,
    padding='same',
    input_shape=(1,28,28)
))
model.add(Activation('relu'))
# pooling压缩 添加pooling池化层
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    padding='same',
))
# 第二层卷积层
model.add(Convolution2D(filters=64,kernel_size=5,padding='same'))
model.add(Activation('relu'))
# 第二层取样池化
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
# 进入全连接层
# 抹平为一维数据
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
# 输出层
model.add(Dense(10))
model.add(Activation('softmax'))
# 定义 optmazier
adam = Adam(lr=1e-4)
# 编译
model.compile(optimizer=adam,
              loss='categorical_crossentropy'
              ,metrics=['accuracy'])

print('-----Train----')
model.fit(X_train,y_train,epochs=1,batch_size=32)
print('--------Test-----------')
loss,accuracy = model.evaluate(X_test,y_test)
print('loss = ',loss)
print('accuracy',accuracy)
