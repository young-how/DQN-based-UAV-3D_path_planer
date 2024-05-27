'''
Author: younghow 1102708501@qq.com
Date: 2023-10-07 09:57:08
LastEditors: younghow 1102708501@qq.com
LastEditTime: 2023-11-07 10:04:03
FilePath: \RLGF\simulator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
###simulator.py
##功能：所有类别的顶端控制器，实现各类别参数的调控，
##并通过参数的调控生成不同的工厂类
import time
from BaseClass.CalMod import *
from FactoryClass.EnvFactory import *
import sys
import numpy as np
#sys.path.append("E:\younghow\RLGF")
import os
from tqdm import tqdm
root=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../config/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Agents') 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
import csv 
import datetime
from BaseClass.BaseCNN import *
cur_time=datetime.datetime.now()
cur_time= cur_time.strftime('%m_%d_%Y(%H_%M_%S)')
import random
# 设置全局随机种子，可以使用任何整数作为种子值
seed_value = 42  # 这里使用42作为种子值，您可以选择任何整数
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#设置所有随机数种子
    

class simulator():
    def __init__(self) -> None:
        #DQN训练相关参数(默认)
        self.TARGET_UPDATE = 10   #Q网络更新周期
        self.num_episodes = 150000  #训练周期长度
        self.print_every = 10 #训练记录周期
        self.min_eps = 0.1   #最小贪心概率
        self.max_eps_episode =1000  #最大贪心次数
        self.EnvFactory=EnvFactory()  #实例化工厂类
        self.epoch=0      #当前训练轮数
        self.env=None           #实例化环境
        self.result_path='logs/score_%s.csv' %cur_time
        self.CsvWriter=None   #csv写入类
        self.Max_score=-9999999999  #记录最高得分

        #仿真模拟控制器信号
        self.stat=1   #1:开始 2：暂停 3：停止

        #自动载入配置文件，进行初始化
        self.Init_From_XML()

        #训练结果记录模块初始化
        self.Init_Record_Mod()

        self.infos=[]   #训练的反馈列表
        #执行时间统计
        self.executed_time=0
        

    def Init_Record_Mod(self):
        #初始化训练记录模块
        file=open(self.result_path,'a+',newline="")
        self.CsvWriter=csv.writer(file)
        #self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','step'])
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','step','avg_trainning_time','avg_testing_time','total_time'])
 
    def Init_From_XML(self,XML_path=root+'/config/FMEC_experience.xml'):
        #根据配置文件初始化模拟器参数,根据工厂类生成 env类
        try:
            config_dict=XML2Dict(XML_path)   #解析xml文件
            if config_dict!=None: 
                config_dict=config_dict.get('simulator')
                self.record_epo=int(config_dict.get('record_epo'))
                self.num_episodes=int(config_dict.get('num_episodes'))
                self.max_eps_episode=int(config_dict.get('max_eps_episode'))
                self.min_eps=float(config_dict.get('min_eps'))
                self.num_episodes=int(config_dict.get('num_episodes'))
                self.TARGET_UPDATE=int(config_dict.get('TARGET_UPDATE'))
                # #记录存放路径
                # if config_dict.get('record_path')!= None:
                #     self.result_path=config_dict.get('record_path')

                #创建环境类
                env_dict=config_dict.get('env')
                self.env=self.EnvFactory.Create_Env(env_dict)  
        except Exception as e:
            print(e.args)  #输出异常信息
            return None

    def StartAndTrain(self,flag=0):
        #开始仿真模拟，并同步进行训练
        #输入参数：
        #       flag    ：      训练方式，0：随机重置环境   1：根据配置文件配置
        # while self.epoch<self.num_episodes:
        #     if self.stat==2:
        #         continue   #暂停
        #     elif self.stat==3:
        #         return      #停止训练与模拟
        #     elif self.stat==1:
        #         if flag==0:
        #             self.epoch+=1
        #             eps_rate=self.epsilon_annealing()       #获取贪心概率
        #             info=self.env.run_eposide(eps_rate)   #计算机随机生成场景进行训练
        #             self.epoch+=1
        #             self.record(info)   #将记录写入文件中
        #             if self.epoch%self.print_every==0:
        #                 #在控制台输出并保存训练信息
        #                 print(info)
        #                 self.record(info)
        #         else:
        #             self.env.run_XML_scene()   #根据现有的场景进行训练

        #带进度条的训练
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes / 10)):
                    if self.stat==2:
                        continue   #暂停
                    elif self.stat==3: 
                        return      #停止训练与模拟
                    elif self.stat==1: 
                        if flag==0:
                            self.epoch+=1
                            eps_rate=self.epsilon_annealing()       #获取贪心概率
                            epside_start=time.time()
                            info=self.env.run_eposide(eps_rate)   #计算机随机生成场景进行训练
                            epside_end=time.time()
                            self.executed_time+=(epside_end-epside_start)
                            self.infos.append(info)    #添加反馈列表
                            if self.Max_score<info['average_score']:
                                self.Max_score=info['average_score']
                            self.record_list()  #对缓存列表进行统计
                        else:
                            self.env.run_XML_scene()   #根据现有的场景进行训练


                        # loss=0
                        # if type(info['loss'])!=int:
                        #     loss=info['loss'].item()
                        #更新进度条
                        if (i_episode + 1) % 2 == 0:
                            pbar.set_postfix({
                                'episode':
                                '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                                'MAX return':
                                '%.3f' % self.Max_score,
                                'return':
                                '%.3f' % info['average_score'],
                                'loss':
                                '%.3f' % info['loss']
                            })
                        pbar.update(1)

    def epsilon_annealing(self):
        #返回贪心概率
        slope = (self.min_eps - 1.0) / (self.max_eps_episode+0.1)
        ret_eps = max(slope * self.epoch + 1.0, self.min_eps)
        return ret_eps        

    def Update_target(self):
        #更新双网络结构的网络参数
        self.env.Trainer.hard_update()

    def record_list(self):
        sum_epoch=score=average_score=eps=success=lose=meet_threaten=loss=step=0
        # list_len=len(self.infos)
        # if self.infos!=[]:
        #     for line in self.infos:
        #         sum_epoch+=line['sum_epoch']
        #         score+=line['score']
        #         average_score+=line['average_score']  
        #         eps+=line['eps']
        #         success+=line['success']
        #         lose+=line['lose']
        #         meet_threaten+=line['meet_threaten']
        #         loss+=abs(line['loss'])   #loss的绝对值
        #         # step+=line['step']
        #         step+=0
        #     self.CsvWriter.writerow([sum_epoch/list_len,self.epoch,score/list_len,average_score/list_len,eps/list_len,success/list_len,lose/list_len,meet_threaten/list_len,loss/list_len,step/list_len])
        self.infos=[]  #清空列表
        traing_time=0
        testing_time=0
        for uav in self.env.Agents:
            traing_time+=uav.Train_time
            testing_time+=uav.Testing_time
        traing_time=traing_time/len(self.env.Agents)
        testing_time=testing_time/len(self.env.Agents)
        self.CsvWriter.writerow([sum_epoch,self.epoch,score,average_score,eps,success,lose,meet_threaten,loss,step,traing_time,testing_time,self.executed_time])

    def record(self,info=None):
        #将运行结果进行记录
        if info!=None:
            self.CsvWriter.writerow([info['sum_epoch'],self.epoch,info['score'],info['average_score'],info['eps'],info['success'],info['lose'],info['meet_threaten'],info['loss']])

    def Sim(self):
        #根据场景配置文件进行仿真模拟
        pass
    def Reset_Env(self):
        #随机重置训练环境
        pass 
    def Reset_Env_XML(self):
        #根据XML重置训练环境
        pass
    #设置单次训练批量大小 
    def Set_Batch_Size(self,bath_size): 
        self.BATCH_SIZE=bath_size
    #设置DQN的折扣率
    def Set_Gamma(self, gamma):
        self.gamma = gamma
    #设置训练学习率
    def Set_Learning_Rate(self, learning_rate):
        self.LEARNING_RATE = learning_rate
    #设置target网络更新周期
    def Set_Target_Update(self, target_update):
        self.TARGET_UPDATE = target_update
    #设置训练周期长度
    def Set_Num_Episodes(self, num_episodes):
        self.num_episodes = num_episodes
    #设置训练记录周期间隔
    def Set_Print_Every(self, print_every):
        self.print_every = print_every
    #设置最小贪心概率 
    def Set_Min_Eps(self, min_eps):
        self.min_eps = min_eps
    #设置最大贪心次数
    def Set_Max_Eps_Episode(self, max_eps_episode):
        self.max_eps_episode = max_eps_episode
    #设置状态空间大小
    def Set_Space_Dim(self, space_dim):
        self.space_dim = space_dim
    #设置动作空间大小
    def Set_Action_Dim(self, action_dim):
        self.action_dim = action_dim
        
if __name__=="__main__":
    sim=simulator()
    print(sim.env)
    sim.StartAndTrain()