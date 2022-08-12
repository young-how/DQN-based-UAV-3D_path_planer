######################################################################
# DQN Model Train
#---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
#对训练参数进行设置，并对基于DQN的无人机航迹规划算法模型进行训练
#----------------------------------------------------------------
# Set the training parameters and train the UAV track planning algorithm model based on DQN
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
from collections import deque
from replay_buffer import ReplayMemory, Transition
from  torch.autograd import Variable
import torch
import torch.optim as optim
import random
from model import QNetwork

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
from  torch.autograd import Variable

from replay_buffer import ReplayMemory, Transition

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#plt.ion()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 128    #批量大小
TAU = 0.005 
gamma = 0.99   #折扣率
LEARNING_RATE = 0.0004   #学习率
TARGET_UPDATE = 10   #Q网络更新周期

num_episodes = 40000  #训练周期长度
print_every = 1  
hidden_dim = 16 ## 64 ## 16
min_eps = 0.01    #贪心概率
max_eps_episode = 10   #最大贪心次数



#env = gym.wrappers.Monitor(env, directory="monitors", force=True)
        
space_dim = 42 # n_spaces   状态空间维度
action_dim = 27 # n_actions   动作空间维度
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

threshold = 200    #？？？？
env = Env(space_dim,action_dim,LEARNING_RATE)
print('threshold: ', threshold)

#agent = Agent(space_dim, action_dim, hidden_dim)  #构造智能体

    
def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1  
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps        

def save(directory, filename):  #存放Q网络参数
    torch.save(env.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(env.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))

def run_episode(env, eps):
    """Play an epsiode and train  进行训练

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action        
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode  返回回报值
    """
    state = env.reset()  #环境重置
    #done = False
    total_reward = 0
    
    #env.render(1)
    n_done=0
    count=0
    success_count=0   #统计任务完成数量
    crash_count=0   #坠毁无人机数量
    bt_count=0   #电量耗尽数量
    over_count=0   #超过最大步长的无人机
    while(1):
        count=count+1
        for i in range(len(env.uavs)):
            if env.uavs[i].done:
                #无人机已结束任务，跳过
                continue
            action = env.get_action(FloatTensor(np.array([state[i]])) , eps)   #根据Q值选取动作
            
            next_state, reward, uav_done, info= env.step(action.detach(),i)  #根据选取的动作改变状态，获取收益

            total_reward += reward  #求总收益
                        
            # Store the transition in memory   存储交互经验
            env.replay_memory.push(
                    (FloatTensor(np.array([state[i]])), 
                    action, # action is already a tensor
                    FloatTensor([reward]), 
                    FloatTensor([next_state]), 
                    FloatTensor([uav_done])))
            """ if reward>0:
                #正奖励，加强经验
                for t in range(2):
                    env.replay_memory.push(
                        (FloatTensor(np.array([state[i]])), 
                        action, # action is already a tensor
                        FloatTensor([reward]), 
                        FloatTensor([next_state]), 
                        FloatTensor([uav_done]))) """
            if info==1:
                success_count=success_count+1
            elif info==2:
                crash_count+=1
            elif info==3: 
                bt_count+=1
            elif info==5: 
                over_count+=1

            if uav_done:   #结束状态
                env.uavs[i].done=True
                n_done=n_done+1
                continue
            state[i] = next_state  #状态变更
        #env.render()
        if count%5==0 and len(env.replay_memory) > BATCH_SIZE:
            #batch = env.replay_memory.sample(BATCH_SIZE) 
            env.learn(gamma,BATCH_SIZE)  #训练Q网络
        if n_done>=env.n_uav:
            break
        #plt.pause(0.001)
    if success_count>=0.8*env.n_uav and env.level<10:
        env.level=env.level+1   #通过率较大，难度升级
    return total_reward,[success_count,crash_count,bt_count,over_count]
def train():    

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    
    
    time_start = time.time()
    #载入预训练模型
    check_point_Qlocal=torch.load('Qlocal.pth')
    check_point_Qtarget=torch.load('Qtarget.pth')
    env.q_target.load_state_dict(check_point_Qtarget['model'])
    env.q_local.load_state_dict(check_point_Qlocal['model'])
    env.optim.load_state_dict(check_point_Qlocal['optimizer'])
    epoch=check_point_Qlocal['epoch']

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)  #计算贪心概率
        score,info = run_episode(env, eps)  #运行一幕，获得得分,返回到达目标的个数

        scores_deque.append(score)  #添加得分
        scores_array.append(score)
        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)
            
        if i_episode % print_every == 0 and i_episode > 0:
            print('sum_Episode: {:5} Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, eps-greedy: {:5.2f} Time: {:02}:{:02}:{:02} level:{:5}  num_success:{:2}  num_crash:{:2}  num_none_energy:{:2}  num_overstep:{:2}'.\
                    format(i_episode+epoch,i_episode, score, avg_score, eps, dt//3600, dt%3600//60, dt%60,env.level,info[0],info[1],info[2],info[3]))
        #保存模型参数
        if i_episode %100==0:
            #每100周期保存一次网络参数
            state = {'model': env.q_target.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qtarget.pth")
            state = {'model': env.q_local.state_dict(), 'optimizer': env.optim.state_dict(), 'epoch': i_episode+epoch}
            torch.save(state, "Qlocal.pth")

        if i_episode % TARGET_UPDATE == 0:
            env.q_target.load_state_dict(env.q_local.state_dict()) 
    
    return scores_array, avg_scores_array

  


if __name__ == '__main__':
    scores,avg_scores=train()
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))
