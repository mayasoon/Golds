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
    state_actions = q_table.iloc[state,:] # 获取state行所有的值
    if (numpy.random.uniform() > epsilon) or (state_actions.all() == 0): # 非贪婪 或者初始探索时随机取一个值
        action_name = numpy.random.choice(actions)
    else:
        action_name = state_actions.argmax() # 贪婪模式选择最大的数值
    return action_name
def get_env_feedback(state,action):# 传入位置和动作信息来判断，下一步如何走
    if action == 'right':# 向右走的话离宝藏近
        if state == n_states - 2:# 到达宝藏位置
            S_ = 'terminal'
            R = 1
        else:
            S_ = state+1
            R = 0
    else:
        R = 0
        if state == 0:
            S_ = state
        else:
            S_ = state-1
    return S_,R
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(n_states-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(fresh_time)
def rl():
    q_table = build_table(n_states,actions)
    for episode in range(max_episodes):# 循环每一幕，场景
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)# 更新环境信息界面显示
        while not is_terminated:
            A = choose_action(S, q_table) # 选行为 right或者left
            S_,R = get_env_feedback(S,A) # 实施行为并得到环境的反馈,反馈值和下一步的位置
            q_predict = q_table.loc[S,A] # 估算的(状态-行为)值 获取第几行第几列的行为值
            if S_ != 'terminal':
                q_target = R+gamma*q_table.iloc[S_,:].max()#  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R # 实际的(状态-行为)值 (回合结束)
                is_terminated = True
            q_table.loc[S,A] +=alpha*(q_target-q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state
            update_env(S,episode,step_counter+1) # 环境更新
            step_counter+=1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print(q_table)