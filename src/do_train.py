import numpy as np
import matplotlib.pyplot
# 训练开始
from src.learning import NeuralNetWork

data_file = open("C:/Users/maya/Desktop/mnist_train.csv","r")
data_list = data_file.readlines()
data_file.close()

# 初始化神经网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

neural = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

epochs = 5
for e in range(epochs):
    for record in data_list:
        all_values = record.split(",")
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.1
        targets[int(all_values[0])] = 0.99
        neural.train(scaled_input,targets)
    pass
# 测试单个训练成果
test_file = open("C:/Users/maya/Desktop/mnist_test.csv","r")
test_list = test_file.readlines()
test_file.close()
# test_all_values = test_list[0].split(",")
# result = neural.query((np.asfarray(test_all_values[1:]) / 255.0 * 0.99) + 0.01)
# print(result)
# 检验准确率
scorecard = []
for record in test_list:
    all_values = record.split(",")
    # 正确的数字
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = neural.query(inputs)
    # 最大值的下标
    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print("准确率",scorecard_array.sum() / scorecard_array.size)
