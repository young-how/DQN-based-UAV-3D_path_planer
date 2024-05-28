#威胁对象基类，所有自定义威胁对象需要继承该类
from CalMod import *
class BaseThreaten():
    def __init__(self,param:dict) -> None:
        #初始化基本位置信息与类型信息
        self.position=Loc(float(param.get("position").get('x')),float(param.get("position").get('y')),float(param.get("position").get('z')))  #初始化坐标位置
        self.type=param.get("type")         #威胁类型

    def check_threaten(self,p:Loc):
        #输入一个坐标点，返回该点是否在该威胁的威胁范围内，或威胁度
        #输入参数：
        #       p   :   待测坐标点
        #输出参数：
        #       re  ：  该威胁对该店的威胁度或该点是否在该威胁区
        pass

    def run(self):
        #动态更新自己的威胁参数
        pass