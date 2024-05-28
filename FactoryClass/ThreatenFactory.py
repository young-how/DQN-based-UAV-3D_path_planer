###威胁对象工厂类，用于创建威胁对象
import sys
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Threatens")
# sys.path.append("E:\younghow\RLGF\BaseClass")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Obstacles')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
import importlib
class ThreatenFactory():
    def Create_Threaten(self,param,env=None):
        #输入定义好的智能体类型，创建该类型的实例并返回
        #参数说明：
        #       Threaten_Type       ：      威胁名称，要与自定义的.py文件名称一致
        #       param               ：      初始化参数，以字典形式存放初始化参数 
        try:
            Threaten_Type=param.get('Threaten_Type')
            module=importlib.import_module(Threaten_Type)  #导入自定义智能体模型
            ThreatenEntity = getattr(module, Threaten_Type)(param,env)  #构造目标类
        except Exception as e:
            print(e.args)  #输出异常信息
            return None
        else:
            return ThreatenEntity