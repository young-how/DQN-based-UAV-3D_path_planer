#神经网络创建工厂，根据输入的神经网络名称创建对应的网络
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Trainer")
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from BaseClass.BaseCNN import *
import importlib
class NetworkFactory():
    def Create_Network(self,param):
        #输入定义好的神经网络名称，创建该网络的实例并返回
        #参数说明：
        #       Trainer_Type        ：      训练器名称，要与自定义的.py文件名称一致
        #       param               ：      初始化参数，以字典形式存放初始化参数 
        try:
            Network_Type=param.get('NetWork')
            module=importlib.import_module('BaseCNN')  #导入基础网络库
            TrainerEntity = getattr(module, Network_Type)(param)  #构造目标神经网络
        except Exception as e:
            print(e.args)  #输出异常信息
            return None
        else:
            return TrainerEntity