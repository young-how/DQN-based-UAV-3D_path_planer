#Agent基类
##继承该类以实现DQN智能体的功能，便于快速拓展与自定义智能体行为
#状态统一使用二维图像形式进行表示
import sys
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Agents")
# sys.path.append("E:\younghow\RLGF\BaseClass")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Agents')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from CalMod import *
import numpy as np
class BaseAgent():
    def __init__(self,param:dict) -> None:
        self.position=Loc(int(param.get("position").get('x')),int(param.get("position").get('y')),int(param.get("position").get('z')))  #初始化在空间中的位置
        #self.Actions=param.get("Actions")  #动作空间
        # self.goal=Loc(int(param.get("goal").get('x')),int(param.get("goal").get('y')),int(param.get("goal").get('z')))   #目标点
        # self.Dis_Start2Goal=Eu_Loc_distance(self.position,self.goal) #距离目标点的距离

        #基本状态图层(可自定义，通过Env_Sensor函数进行创建赋值)
        self.state_map=None  #self.state() 函数最终返回的状态图
        #可自定义的状态图层示例如下：
        # self.APF_map=None   #人工势场状态图，可建立目标点对智能体的引力模型
        # self.Ob_map=None   #障碍物图层，可描述空间的可行性关系
        # self.Route_map=None #路线图层，可用来记录走过的路线或者通过辅助路线加强学习效果
    
        #强化学习交互部分
        self.done=False
        self.score=0   #智能体交互得分
        
    def Set_State_Map(self,L:int,W:int,H:int):
        #设置状态图层形状
        #L图层长度，W图层宽度，H图层厚度
        self.state_map=np.zeros((1,H,L,W))

    def Set_Actions(self,Action_Dic):
        #通过设定好的动作集合初始化智能体的动作空间
        #Action_Dic为动作空间字典，包含了对应动作编号下的坐标（或姿态）变更规则
        #例如actions={'L':Loc(0,-1,0),'R':Loc(0,1,0),'U':Loc(1,0,0),'L':Loc(-1,0,0)}
        self.Actions=Action_Dic
    
    def Observation(self,env):
        #该函数实现环境状态感知，使智能体能感知到环境类中的部分或全部成员
        pass

    def reset(self):
        #该函数使智能体回到初始状态
        pass
    
    def get_action(self,action):
        #该函数给定动作编号，使智能体完成该动作，从而更新并返回智能体状态
        pass
    
    def state(self):
        #返回智能体的状态，注意与神经网络定义的输入一致
        pass