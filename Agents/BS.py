from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
import random
import math
#构建基站智能体
#问题描述：
#每个BS在每个时间步，统计半径内的UE，能够采集的
class BS(BaseAgent):
    def __init__(self,p:Loc) -> None:
        self.num_UE=0   #UE的数目
        self.um_task=0   #任务数目
    
    def run(self):
        #位置移动
        x=self.position.x+math.cos(random.uniform(0,2*math.pi))*self.step_R
        y=self.position.x+math.sin(random.uniform(0,2*math.pi))*self.step_R
        if x<0:
            x=0
        elif x>100:
            x=100
        if y<0:
            y=0
        elif y>100:
            y=100
        self.position.x=x
        self.position.y=y

        #作业收集
        self.task+=self.speed_collect  #作业收集
        if self.task>self.Max_task:
            #等待时间
            self.Caculate_wait_time+=1
            self.task=self.Max_task
        elif self.task>3:
            self.flag=1   #是否需要计算

    #状态重置，已经被无人机收集过了
    def reset(self):
        self.flag=0   #是否需要计算
        self.task=0   #收集的作业数目
        self.Caculate_wait_time=0   #作业处理等待时间