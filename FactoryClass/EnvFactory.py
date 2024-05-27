#环境对象工厂类，创建自定义环境类
import sys
import os
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Envs")
# sys.path.append("E:\younghow\RLGF\BaseClass")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Envs')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from BaseClass.CalMod import *
import importlib
class EnvFactory():
    def Create_Env(self,param):
        #输入定义好的智能体类型，创建该类型的实例
        #参数说明：
        #       Env_Type            ：      环境名称，要与自定义的.py文件名称一致
        #       param               ：      初始化参数，以字典形式存放初始化参数 
        try:
            Env_Type=param.get('Env_Type')
            module=importlib.import_module(Env_Type)  #导入自定义智能体模型
            EnvEntity = getattr(module, Env_Type)(param)  #构造目标类
        except Exception as e:
            print(e.args)  #输出异常信息
            return None
        else:
            return EnvEntity

if __name__=='__main__':
    Fc=EnvFactory()
    start=Loc(1,2,3)
    dic={"position":{'x':'1','y':'1','z':'1'},"Env_Type":'KZ_env'}  #初始化字典
    zdj=Fc.Create_Env(dic)
    print(zdj)