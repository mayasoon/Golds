import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten

import matplotlib.pyplot as plt

X = np.linspace(-1,1,300)
# 打乱顺序
np.random.shuffle(X)
# 用待测函数生成 Y值，正态分布随机生成后面的值 作为一个随机加减项
Y = 0.5*X+2+np.random.normal(0,0.05,300)
# plot data
# plt.scatter(X, Y)
# plt.show()

# 取前160个为训练数据 ，后40个位测试数据
X_train,Y_train = X[:260],Y[:260]
X_test,Y_test = X[len(X)-40:],Y[len(Y)-40:]


# 模型层
model = Sequential()
# 添加隐藏层
model.add(Dense(input_dim=1, units=1,use_bias=True))
model.compile(loss='mse',optimizer='sgd')

print('---train---')
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print('train cost = ',cost)

# test 测试训练结果 误差值
print('----test----')
test_cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test_cost = ' ,test_cost)
W,b = model.get_weights()
print('weights ' ,W,' bb ' ,b)