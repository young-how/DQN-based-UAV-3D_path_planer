######################################################################
# Environment build
#---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
#env类对城市环境进行三维构建与模拟，利用立方体描述城市建筑，
# 同时用三维坐标点描述传感器。对环境进行的空间规模、风况、无人机集合、
# 传感器集合、建筑集合、经验池进行初始化设置，并载入DQN神经网络模型。
# env类成员函数能实现UAV行为决策、UAV决策经验学习、环境可视化、单时间步推演等功能。
#----------------------------------------------------------------
# The env class constructs and simulates the urban environment in 3D, 
# uses cubes to describe urban buildings, and uses 3D coordinate points 
# to describe sensors. Initialize the environment's spatial scale, wind conditions,
# UAV collection, sensor collection, building collection, and experience pool, 
# and load the DQN neural network model. 
# The env class member function can implement UAV behavioral decision-making,
# UAV decision-making experience learning, environment visualization, and single-time-step deduction.
##############################################################################
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import random
from model import QNetwork
from UAV import *
from  torch.autograd import Variable
from replay_buffer import ReplayMemory, Transition
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
class building():
    def __init__(self,x,y,l,w,h):
        self.x=x   #建筑中心x坐标
        self.y=y   #建筑中心y坐标
        self.l=l        #建筑长半值
        self.w=w       #建筑宽半值
        self.h=h    #建筑高度
class sn():
    def __init__(self,x,y,z):
        self.x=x 
        self.y=y 
        self.z=z
class Env(object):
    def __init__(self,n_states,n_actions,LEARNING_RATE):
        #定义规划空间大小
        self.len=100
        self.width=100
        self.h=22
        self.map=np.zeros((self.len,self.width,self.h))
        self.WindField=[30,0]     #风场(风速,风向角)
        #self.action_space=spaces.Discrete(27)  #定义无人机动作空间(0-26),用三进制对动作进行编码 0:-1 1：0 2：+1
        #self.observation_space=spaces.Box(shape=(self.len,self.width,self.h),dtype=np.uint8)  #定义观测空间（规划空间）,能描述障碍物情况与风向
        self.uavs=[]  #无人机对象集合
        self.bds=[]   #建筑集合
        self.target=[]  #无人机对象
        self.n_uav=15   #训练环境中的无人机个数
        self.v0=40  #无人机可控风速
        self.fig=plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        plt.ion()  #interactive mode on
        self.level=1    #训练难度等级(0-10)

        #神经网络参数
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)   #初始化Q网络
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)   #初始化目标Q网络
        self.mse_loss = torch.nn.MSELoss()     #损失函数：均方误差
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)   #设置优化器，使用adam优化器
        self.n_states = n_states     #状态空间数目？
        self.n_actions = n_actions    #动作集数目

        #  ReplayMemory: trajectory is saved here
        self.replay_memory = ReplayMemory(10000)   #初始化经验池

    def get_action(self, state, eps, check_eps=True):
        """Returns an action  返回行为值

        Args:
            state : 2-D tensor of shape (n, input_dim)
            eps (float): eps-greedy for exploration  eps贪心策略的概率

        Returns: int: action index    动作索引
        """
        global steps_done
        sample = random.random()

        if check_eps==False or sample > eps:
            with torch.no_grad():
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)   #根据Q值选择行为
        else:
           ## return LongTensor([[random.randrange(2)]])
           return torch.tensor([[random.randrange(self.n_actions)]], device=device)   #随机选取动作
    def learn(self, gamma,BATCH_SIZE):
        """Prepare minibatch and train them  准备训练

        Args:
        experiences (List[Transition]): batch of `Transition`   
        gamma (float): Discount rate of Q_target  折扣率
        """
        
        if len(self.replay_memory.memory) < BATCH_SIZE:
            return
            
        transitions = self.replay_memory.sample(BATCH_SIZE)  #获取批量经验数据
        
        batch = Transition(*zip(*transitions))
                        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        
            
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to newtork q_local (current estimate)
        Q_expected = self.q_local(states).gather(1, actions)     #获得Q估计值

        Q_targets_next = self.q_target(next_states).detach().max(1)[0]   #计算Q目标值估计

        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))   #更新Q目标值
        #训练Q网络
        self.q_local.train(mode=True)        
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))  #计算误差
        # backpropagation of loss to NN        
        loss.backward()
        self.optim.step()
               
        
    def soft_update(self, local_model, target_model, tau):
        """ tau (float): interpolation parameter"""
        #更新Q网络与Q目标网络
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)     
            
    def hard_update(self, local, target):
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(param.data)
    def render(self,flag=0):
        #绘制封闭立方体
        #参数
        #x,y,z立方体中心坐标
        #dx,dy,dz 立方体长宽高半长
        #fig = plt.figure()
        if flag==1:
            #第一次渲染，需要渲染建筑
            z=0
            #ax = self.fig.add_subplot(1, 1, 1, projection='3d')
            for ob in self.bds:
                #绘画出所有建筑
                x=ob.x
                y=ob.y
                z=0
                dx=ob.l 
                dy=ob.w 
                dz=ob.h 
                xx = np.linspace(x-dx, x+dx, 2)
                yy = np.linspace(y-dy, y+dy, 2)
                zz = np.linspace(z, z+dz, 2)

                xx2, yy2 = np.meshgrid(xx, yy)

                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
            

                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x-dx), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)

                xx2, zz2= np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y-dy), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)
            for sn in self.target:
                #绘制目标坐标点
                self.ax.scatter(sn.x, sn.y, sn.z,c='red')
        
        for uav in self.uavs:
            #绘制无人机坐标点
            self.ax.scatter(uav.x, uav.y, uav.z,c='blue')


    def step(self, action,i):
        """环境的主要驱动函数，主逻辑将在该函数中实现。该函数可以按照时间轴，固定时间间隔调用

        参数:
            action (object): an action provided by the agent
            i:i号无人机执行更新动作

        返回值:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """
        reward=0.0
        done=False
        #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=0
        reward,done,info=self.uavs[i].update(action)  #无人机执行行为,info为是否到达目标点
        #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
        next_state = self.uavs[i].state()
        return next_state,reward,done,info
    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        #重置画布
        #plt.close()
        #self.fig=plt.figure()
        #self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        #plt.clf()
        self.uavs=[]
        self.bds=[]
        self.map=np.zeros((self.len,self.width,self.h))  #重置障碍物
        self.WindField=[]
        #生成随机风力和风向
        self.WindField.append(np.random.normal(40,5))
        self.WindField.append(2*math.pi*random.random())
        """ #设置边界障碍物
        self.map[0:self.len,0:self.width,0]=1
        self.map[0:self.len,0:self.width,self.h-1]=1
        self.map[0:self.len,0:self.width,0:self.h]=1
        self.map[self.len-1,0:self.width,0:self.h]=1
        self.map[0:self.len,0,0:self.h]=1
        self.map[0:self.len,self.width-1,0:self.h]=1 """
         #随机生成建筑物
        for i in range(random.randint(self.level,2*self.level)):
            self.bds.append(building(random.randint(10,self.len-10),random.randint(10,self.width-10),random.randint(1,10),random.randint(1,10),random.randint(9,13)))
            self.map[self.bds[i].x-self.bds[i].l:self.bds[i].x+self.bds[i].l,self.bds[i].y-self.bds[i].w:self.bds[i].y+self.bds[i].w,0:self.bds[i].h]=1

        #随机生成目标点位置
        x=0
        y=0
        z=0
        while(1):
            x=random.randint(60,90)
            y=random.randint(10,90)
            z=random.randint(3,15)
            if self.map[x,y,z]==0:
                #随机生成在无障碍区域
                break
        self.target=[sn(x,y,z)]
        self.map[x,y,z]=2

        #随机生成无人机位置
        for i in range(self.n_uav):
            x=0
            y=0
            z=0
            while(1):
                x=random.randint(15,30)
                y=random.randint(10,90)
                z=random.randint(3,7)
                if self.map[x,y,z]==0:
                    #随机生成在无障碍区域
                    break
            self.uavs.append(UAV(x,y,z,self))  #重新生成无人机位置
            #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

        #更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state
    def reset_test(self):
        #环境重组测试
        self.uavs=[]
        self.bds=[]
        self.map=np.zeros((self.len,self.width,self.h))  #重置障碍物
        self.WindField=[]
        #生成随机风力和风向
        self.WindField.append(np.random.normal(40,5))
        self.WindField.append(2*math.pi*random.random())
         #随机生成建筑物
        for i in range(random.randint(self.level,2*self.level)):
            self.bds.append(building(random.randint(10,self.len-10),random.randint(10,self.width-10),random.randint(1,10),random.randint(1,10),random.randint(9,13)))
            self.map[self.bds[i].x-self.bds[i].l:self.bds[i].x+self.bds[i].l,self.bds[i].y-self.bds[i].w:self.bds[i].y+self.bds[i].w,0:self.bds[i].h]=1

        #随机生成目标点位置
        x=0
        y=0
        z=0
        while(1):
            x=random.randint(60,90)
            y=random.randint(10,90)
            z=random.randint(3,15)
            if self.map[x,y,z]==0:
                #随机生成在无障碍区域
                break
        self.target=[sn(x,y,z)]
        self.map[x,y,z]=2

        #生成无人机位置
        self.uavs.append(UAV(20,20,3,self))  #重新生成无人机位置
            #self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
       
        #更新无人机状态
        self.state=np.vstack([uav.state() for (_, uav) in enumerate(self.uavs)])

        return self.state

if __name__ == "__main__":
    env=Env()
  
    env.reset()
    env.render()
    plt.pause(30)

