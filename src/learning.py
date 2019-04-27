import numpy as np
import scipy.special as sp

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate
        # w11 ,w21
        # w12 ,w22 etc 使用正态概率分布采样权重，其中平均值为0，标准方差为节点传入链接数目的开放分之1即 1/√传入链接数目
        self.wih = np.random.normal(0.0,pow(self.innodes,-0.5),(self.hnodes,self.innodes))
        self.who = np.random.normal(0.0,pow(self.innodes,-0.5),(self.outnodes,self.hnodes))
        # 激活函数
        self.activation_function = lambda x: sp.expit(x)

        pass
    # 反向传播更改权重值
    def train(self,input_list,target_list):
        # 把目标数组变成所需的格式的矩阵 竖着的数组
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        # 隐藏层的输出值
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        # 最终输出值
        final_output = self.activation_function(final_inputs)

        # 计算目标值于实际输出值之间的误差
        output_errors = targets - final_output
        # 隐藏层的误差值矩阵
        hidden_errors = np.dot(self.who.T,output_errors)
        # 更新输出层和隐藏层的权重transpose 和T有区别？在就一列或者一行的情况下
        # 先计算出斜率也就是导数值 为 最终层的误差 * 最终层的输出值 * （1 - 最终层输出值）* 上一层的在这里也就是隐藏层的值
        self.who += self.lr * np.dot((output_errors * final_output * (1-final_output)),np.transpose(hidden_outputs))
        # 跟新输入层和隐层中间层的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),np.transpose(inputs))

        pass
    # 计算整个网络得到的实际输出值，作为检查训练结果的方法
    def query(self,input_list):
        # 首先把输入的一维数组转换为 2 维数组 在取逆变为竖着的矩阵
        inputs = np.array(input_list,ndmin=2).T
        # 中间隐藏层的值
        hidden_inputs = np.dot(self.wih,inputs)
        # 隐藏层经过激活函数后的值
        hidden_output = self.activation_function(hidden_inputs)
        # 最后输出层
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)
        return final_output

if __name__ == '__main__':
    pass