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

    def train(self):

        pass
    # 计算整个网络得到的实际输出值
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
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    neural = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    print(neural.query([1.0,0.5,-1.5]))
