#基础环境类，可以继承该类进行进一步开发，添加环境交互规则等
#环境类包含内容：
#AgentFactory类、ThreatenFactory类、与DQN_Factory类
#环境类有独特的环境交互规则，决定了智能体决策的得分与回报
from CalMod import *
from FactoryClass.AgentFactory import *
from FactoryClass.ThreatenFactory import *
from FactoryClass.TrainerFactory import *
import torch
import numpy as np
import math
import random
import csv

from  torch.autograd import Variable
class BaseEnv():
    def __init__(self,param:dict) -> None:
        #构造函数定义环境的空间大小,默认100x100x20
        self.len=int(None2Value(param.get("len"),100))
        self.width=int(None2Value(param.get("width"),100))
        self.h=int(None2Value(param.get("h"),20))

        #环境构成要素
        self.Agents=[]   #智能体集合
        self.Threatens=[]  #威胁对象集合
        self.Trainer=None   #强化学习训练器

        #工厂类
        self.AgentFactory=AgentFactory()   #智能体工厂类
        self.ThreatenFactory=ThreatenFactory()  #威胁工厂类
        self.TrainerFactory=TrainerFactory()    #训练器工厂类

        #是否是AC框架
        self.Is_AC=int(None2Value(param.get("Is_AC"),0))

        
    def Scene_Random_Reset(self):
        #随机重置环境中智能体、威胁的状态，用于强化学习训练的场景生成
        pass
    def Load_Scene_FromXML(self,XML_path="../config/simulator_conf.xml"):
        #从xml文件中对环境成员进行配置，用于特定场景的强化学习训练或者训练效果测试
        #解析XML文件,从XML文件中解析成dict
        p_dict=XML2Dict(XML_path)
        if p_dict!=None:
            env_dict=p_dict.get('env')
            if env_dict!=None:
                #根据参数初始化Agent
                agents_params=env_dict.get('Agents')
                if agents_params != None:
                    agents=agents_params.get('Agent')
                    for agent_param in agents:
                        obj_agent=self.AgentFactory.Create_Agent(agent_param,self)
                        if obj_agent != None:
                            self.Agents.append(obj_agent)
                
                #根据参数初始化Threaten
                threaten_params=env_dict.get('Threatens')
                if threaten_params != None:
                    Threatens=threaten_params.get('Threaten')
                    for threaten_param in Threatens:
                        obj_threaten=self.ThreatenFactory.Create_Threaten(threaten_param,self)
                        if obj_threaten != None:
                            self.Threatens.append(obj_threaten)
        
                #根据参数初始化训练器
                trainer_params=env_dict.get('Trainer')
                if trainer_params != None:
                    self.Trainer=self.TrainerFactory.Create_Trainer(trainer_params)
        
    def Set_Space_Scale(self,len:int,width:int,h:int):
        #重新设置空间大小
        self.len=len
        self.width=width
        self.h=h

    def Check_ZDJ_Done(self):
        #检查环境中的所有智能体是否都进入任务完成状态
        #输出参数：
        #       flag    ：  状态1：所有Agent都完成任务  0：存在Agent没有完成任务
        for agent in self.Agents:
            if not agent.done:
                return False
        return True
    
    def check_threaten(self,p:Loc):
        #给定坐标位置，返回威胁度
        pass

    def Choose_Action(self,index:int,eps=0.2):
        #epsilon贪心策略选取动作索引
        #输入参数
        #       index   ：  智能体所在Agents集合的索引
        #       eps     ：  eps贪心概率
        #输出参数
        #       action  ：  动作值
        if self.Is_AC:
            #actor-critic框架
            sample = random.random()
            if  sample > eps:
                state = self.Agents[index].state()  #指定智能体的状态值
                tensor_state=FloatTensor(np.array([state]))
                probs = self.Trainer.actor(tensor_state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                return action
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(8)]], device=device)   #随机选取动作
        else:
            #DQN框架类型
            state=self.Agents[index].state()  #指定智能体的状态值
            tensor_state=FloatTensor(np.array([state]))
            sample = random.random()
            if  sample > eps:
                with torch.no_grad():
                    y=self.Trainer.q_local(Variable(tensor_state).type(FloatTensor))
                    value=y.data.max(0)[1].view(1, 1) 
                    return  value   #根据Q值选择行为
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(8)]], device=device)   #随机选取动作

    def Move_Agent(self,index:int,action:int):
        #对指定智能体下达移动指令
        #输入参数
        #       index   ：  智能体所在Agents集合中的索引
        #       action  :   对智能体下达的动作编号
        #输出参数
        #       next_state   ：  智能体执行过动作后的状态
        #       reward       :   智能体执行动作后的奖励
        #       done        ：  是否完成任务
        #       info        ：  附加信息
        reward=0.0
        done=False
        reward,done,info=self.Agents[index].update(action)  #智能体执行行为,info为是否到达目标点
        next_state = self.Agents[index].state()
        return next_state,reward,done,info

    def run_random_scene(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #计算机随机生成场景
        pass
    def run_XML_scene(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #根据XML生成场景
        pass 
    def train(self):
        #根据replay_buffer中的数据进行训练
        re=self.Trainer.learn()
        return re
    
    def train_off_policy(self):
        #离线训练（有经验回放池）
        re=self.Trainer.learn_off_policy()
        return re
    
    def train_on_policy(self,transition_dict):
        #在线训练（没有经验回放池）
        re=self.Trainer.learn_on_policy(transition_dict)
        return re
    
    def render(self):
        #可视化接口，利用已有信息进行环境可视化
        pass
    
    def store_experience(self,replay):
        #存放经验信息
        self.Trainer.Push_Replay(replay)