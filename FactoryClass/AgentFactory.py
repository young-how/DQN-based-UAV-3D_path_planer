###智能体工厂类，用于创建智能体对象
import sys
# sys.path.append("E:\younghow\RLGF")
# sys.path.append("E:\younghow\RLGF\Agents")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Agents')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from BaseClass.CalMod import *
import importlib
class AgentFactory():
    def Create_Agent(self,param,env=None):
        #输入定义好的智能体类型，创建该类型的实例
        #输入参数：
        #       Agent_Type          ：      智能体名称，要与自定义的.py文件名称一致
        #       param               ：      初始化参数，以字典形式存放初始化参数 
        #输出参数：
        #       Target_Agent        ：      欲创建的目标对象
        
        try:
            Agent_Type=param.get('Agent_Type')
            module=importlib.import_module(Agent_Type)  #导入自定义智能体模型
            AgentEntity = getattr(module, Agent_Type)(param,env)  #构造目标类
        except Exception as e:
            print(e.args)  #输出异常信息
            return None
        else:
            return AgentEntity

        # Agent_Type=param.get('Agent_Type')
        # module=importlib.import_module(Agent_Type)  #导入自定义智能体模型
        # AgentEntity = getattr(module, Agent_Type)(param,env)  #构造目标类
        # return AgentEntity
        

#test
if __name__=='__main__':
    Fc=AgentFactory()
    start=Loc(1,2,3)
    dic={"position":{'x':'1','y':'1','z':'1'},"Agent_Type":'ZDJ'}  #初始化字典
    zdj=Fc.Create_Agent(dic)
    print(zdj)
    pass
