#训练器创建工厂，根据输入的训练器名称创建对应的算法训练器
import sys
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Trainer")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Trainer')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
import importlib
class TrainerFactory():
    def Create_Trainer(self,param):
        #输入定义好的智能体类型，创建该类型的实例并返回
        #参数说明：
        #       Trainer_Type        ：      训练器名称，要与自定义的.py文件名称一致
        #       param               ：      初始化参数，以字典形式存放初始化参数 
        try:
            Trainer_Type=param.get('Trainer_Type')
            module=importlib.import_module(Trainer_Type)  #导入自定义智能体模型
            TrainerEntity = getattr(module, Trainer_Type)(param)  #构造目标类
        except Exception as e:
            print(e.args)  #输出异常信息
            return None
        else:
            return TrainerEntity