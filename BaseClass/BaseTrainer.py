#基础训练器类，根据此类可进行拓展，成为自定义训练器
import sys
sys.path.append("D:\研究生\RLFG\RLGF(实验专用)\BaseClass")
import torch
import torch.optim as optim

from  torch.autograd import Variable
from BaseClass.CalMod import *
from replay_buffer import *

# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Envs")
# sys.path.append("E:\younghow\RLGF\BaseClass")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Envs')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from FactoryClass.NetworkFactory import *
import csv
class BaseTrainer():
    def __init__(self,param:dict) -> None:
        #训练输入输出相关
        self.h=int(None2Value(param.get('h'),1)) #输入数据高度大小
        self.w=int(None2Value(param.get('w'),1))  #输入数据宽度大小
        self.channel=int(None2Value(param.get('channel'),1))  #输入数据通道数
        self.output=int(None2Value(param.get('output'),1))  #输出个数
        
        
        #获取训练器名称
        self.name=param.get('name')
        
        #训练超参数
        self.replay_size=int(None2Value(param.get('replay_size'),1000))           #经验池大小
        self.LEARNING_RATE=float(None2Value(param.get('LEARNING_RATE'),0.001))           #学习率
        self.Batch_Size=int(None2Value(param.get('Batch_Size'),128))             #批量大小
        self.gamma=float(None2Value(param.get('gamma'),0.99))                           #折扣率
        self.max_epoch=int(None2Value(param.get('max_epoch'),100000))           #最大训练周期数
        self.save_loop=int(None2Value(param.get('save_loop'),10))          #模型参数保存周期

        self.replay_memory=ReplayMemory(self.replay_size)  #初始化经验池
        self.mse_loss = torch.nn.MSELoss()     #损失函数：均方误差
        #self.optim = optim.Adam(self.q_local.parameters(), lr=self.LEARNING_RATE)   #设置优化器，使用adam优化器
        
        #神经网络工厂类
        self.NetworkFactory=NetworkFactory()

        #是否训练
        self.Is_Train=int(None2Value(param.get("Is_Train"),1))
    

    def train_off_policy(self):
        #离线训练（有经验回放池）
        re=self.Trainer.learn_off_policy()
        return re
    
    def train_on_policy(self,transition_dict):
        #在线训练（没有经验回放池）
        re=self.Trainer.learn_on_policy(transition_dict)
        return re

    def learn(self):
        #进行训练
        pass
    def sof_update(self):
        #更新目标网络
        pass
    
    def Push_Replay(self,Experience):
        #将经验存放到经验池中
        self.replay_memory.push(Experience)

    def set_replay_size(self,replay_size:int):
        #设置经验池大小
        self.replay_size=replay_size

    def set_LEARNING_RATE(self, LEARNING_RATE: float):
        # 设置学习率
        self.LEARNING_RATE = LEARNING_RATE

    def set_Batch_Size(self, Batch_Size: int):
        # 设置批量大小
        self.Batch_Size = Batch_Size

    def set_gamma(self, gamma: float):
        # 设置折扣率
        self.gamma = gamma

    def set_max_epoch(self, max_epoch: int):
        # 设置最大训练周期数
        self.max_epoch = max_epoch

    def set_save_loop(self, save_loop: int):
        # 设置模型参数保存周期
        self.save_loop = save_loop