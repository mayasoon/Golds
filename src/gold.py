import numpy
import pandas
import time

# 直线的总长度
n_states = 6
actions = ['left', 'right'] # 探索者的可用动作
epsilon = 0.9 # 贪婪度
alpha = 0.1 # 学习率
gamma = 0.9 # 奖励递减值
max_episodes = 13 # 最大回合数
fresh_time = 0.3 # 没补移动间隔时间

def build_table(state, action):
    table = pandas.DataFrame(# data=None, index=None, columns=None
        numpy.zeros((state, len(action))),# numpy 初始化数据全为零，6 行 两列
        columns=actions # 列的名称
    )
    return table

# print(build_table(n_states, actions))
# 在某个位置的行为动作的选择的方法
def choose_action(state,q_table):
    print('')

if __name__ == '__main__':
    build_table(n_states, actions)

