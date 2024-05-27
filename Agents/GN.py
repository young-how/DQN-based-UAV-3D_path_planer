from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
import random
#地面设备智能体
class GN(BaseAgent):
    def __init__(self,param:dict) -> None:
        self.data=int(param['data'])  #数据容量
        self.max_data=int(param['max_data'])  #数据容量
        self.position=param['position']   #地面节点位置
        self.speed_collect=param['speed_collect']  #数据采集速度
    
    def run(self):
        #正常运行
        self.data=min(self.data+self.speed_collect,self.max_data)  #收集数据
    
    def trans(self,data_num):
        #传输data_num数据出去
        trans_num=self.data
        
        self.data=max(0,self.data-data_num)
        return trans_num-self.data