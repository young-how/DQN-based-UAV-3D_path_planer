import datetime
import json
import time
from tkinter import Grid
from BaseClass.BaseEnv import *
from BaseClass.CalMod import *
import torch
import numpy as np
import math
import random
import csv
import threading
import copy
import heapq
from pyecharts.charts import Surface3D
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
from  torch.autograd import Variable
import pyecharts.options as opts #pip install pyecharts
from pyecharts.charts import Line3D,Scatter3D,Grid
import csv
import uuid
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
from  torch.autograd import Variable

from DataBase.Connector import MySQLConnector
class PathPlan_City(BaseEnv):
    def __init__(self,param:dict) -> None:
        #初始化
        #输入参数：
        #       param   :   初始化参数字典
        super().__init__(param)   #初始化基类
        self.eps=float(None2Value(param.get('eps'),0.1))      #贪心概率
        self.Is_On_Policy=int(param.get('Is_On_Policy'))      #样本采样方式1：在线策略 2：离线策略
        self.param=param
        #威胁描述图层
        self.map=np.zeros((self.len,self.width))   #道路图层
        #房屋建筑
        self.buildings=[]
        threaten_params=param.get('Obstacles')
        if threaten_params != None:        
            buildings_param=os.path.normpath(threaten_params.get('buildings'))
            threaten_config_dict=XML2Dict(buildings_param)    
        self.buildings_param=threaten_config_dict.get('buildings')   
        if self.buildings_param!=None:
            for param_threaten in self.buildings_param["Threaten"] :
                obj_threaten=self.ThreatenFactory.Create_Threaten(param_threaten)
                self.buildings.append(obj_threaten)

        #根据参数初始化Agent
        self.num_UAV=int(param.get('num_UAV'))  #UAV数目
        agents_params=param.get('Agent')
        agents_xml_path=os.path.normpath(agents_params['xml_path_agent'])     
        config_dict=XML2Dict(agents_xml_path)
        uav_params=config_dict.get('Agent')
        for i in range(self.num_UAV):
            uav_params['name']='UAV_'+str(i)
            uav_params['j']=i
            obj_agent=self.AgentFactory.Create_Agent(uav_params,self)
            trainer_xml_path=agents_params['Trainer']
            trainer_xml_path=os.path.normpath(trainer_xml_path.get('Trainer_path'))
            trainer_config_dict=XML2Dict(trainer_xml_path)
            agent_trainer=trainer_config_dict.get('Trainer')
            agent_trainer['name']=obj_agent.name  
            obj_agent.Trainer=self.TrainerFactory.Create_Trainer(agent_trainer)   #创建智能体专属的训练器
            self.Agents.append(obj_agent)
        #仿真推演结果读入与录入
        self.result={'success':0,'failed:':0,'meet_threaten':0,'normal':0,'loss':None,'sum_epoch':0,'eps':0.1} 

        #实验数据记录周期
        self.epoch=0
        self.print_loop=int(param.get('print_loop'))

        #是否采用联邦学习
        self.Is_FL=int(param.get('Is_FL'))
        self.FL_Loop=int(param.get('FL_Loop'))

        #多线程优化
        self.threads=[]
        #执行时间统计
        self.executed_time=0
        #数据库连接类
        DB_path=param.get('DB').get("DB_path")  
        DB_xml_path=os.path.normpath(DB_path)
        DB_config_dict=XML2Dict(DB_xml_path)
        SQL_param=DB_config_dict.get('DB')
        self.SQL_connector=MySQLConnector(SQL_param)
        self.Train_info_line= {
            'uuid': '12345',
            'date': '2024-04-02',
            'html_dir': '<html lang="en"></html>',
            'train_score': 0.75,
            'epoch': 10,
            'Is_Scored': 1,
            'train_time': 123.45,
            'path_len': 50.0,
            'score_quality': 3,
            'score_len': 4,
            'score_threaten': 2
        }
    def getEnvJson(self):
        #生成环境的json字符串
        environment=copy.copy(self.buildings_param)
        environment['path']=[]
        environment['epoch']=self.epoch
        for indx,p in enumerate(self.Agents[0].path):
            environment['path'].append(p)
        json_str = json.dumps(environment)  #将房屋参数进行序列化
        return json_str

    def render(self,data=None):
        init_opts,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3d_opts=self.set_grid() #地图参数配置
        for indx,uav in enumerate(self.Agents):
            path_data=uav.path
        path=self.draw_path(path_data,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3d_opts,init_opts) 
        threaten_3D=self.draw_threaten_3D(None,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3d_opts,init_opts) 
        grid = Grid() #创建空图
        grid.add(path, grid_opts=opts.GridOpts())  
        grid.add(threaten_3D, grid_opts=opts.GridOpts()) 
        html=grid.render_embed()  
        path_json=self.getEnvJson() 
        data2={
            'uuid':data['uuid'],
            'json':path_json  
        }
        save_path='DataBase/experience/'+data['uuid']+'.html'  
        data['html_dir']=save_path   
        grid.render(save_path)
        if self.SQL_connector.connection!=None:
            #连接存在，将训练情况写入DB中
            self.SQL_connector.insert_data('Train_info',data)  
            self.SQL_connector.insert_data('TrainInfo_Json',data2)  
    def draw_threaten_3D(self,marked_points,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3d_opts,init_opts):
        #创建3D威胁图
        re=Surface3D()
        for th in self.buildings:
             (
            re
            .add(
                series_name="",
                shading="color",
                data=list(surface3d_data(th)),
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
                itemstyle_opts=opts.ItemStyleOpts(
                    color='red',  # 设置点的颜色为红色
                    opacity=0.7,  # 可以设置透明度
                ),
            )
        )
        return re
    def set_grid(self):
        #提取ui的xml配置
        ui_param=self.param.get('UI')
        ui_param_path=os.path.normpath(ui_param.get('UI_path'))
        ui_all_dict=XML2Dict(ui_param_path)
        self.ui_dict=ui_all_dict.get('UI')

        #定义显示大小
        InitOpts_dict=self.ui_dict.get('InitOpt')
        init_opts=opts.InitOpts(width=InitOpts_dict.get('width'), height=InitOpts_dict.get('height'))

        self.x_axis3d_opts_dict=self.ui_dict.get('x_axis3d_opts')
        x_axis3d_opts = opts.Axis3DOpts(
            type_="value",
            max_=self.x_axis3d_opts_dict.get('MAX'),  # X轴最大值
            min_=self.x_axis3d_opts_dict.get('MIN'),  # X轴最小值
            split_number=self.x_axis3d_opts_dict.get('split_number')
        )

        self.y_axis3d_opts_dict=self.ui_dict.get('y_axis3d_opts')
        y_axis3d_opts = opts.Axis3DOpts(
            type_="value",
            max_=self.y_axis3d_opts_dict.get('MAX'),  
            min_=self.y_axis3d_opts_dict.get('MIN'),  
            split_number=self.y_axis3d_opts_dict.get('split_number')
        )

        self.z_axis3d_opts_dict=self.ui_dict.get('z_axis3d_opts')
        z_axis3d_opts=opts.Axis3DOpts(
            type_="value",
            max_= self.z_axis3d_opts_dict.get('MAX'), 
            min_= self.z_axis3d_opts_dict.get('MIN'), 
            split_number=self.z_axis3d_opts_dict.get('split_number')
        )

        # 定义网格配置
        self.grid3dopts_dict=self.ui_dict.get('Grid3DOpt')
        grid3dopts=opts.Grid3DOpts(
            width=self.grid3dopts_dict.get('width'), 
            height=self.grid3dopts_dict.get('height'), 
            depth=self.grid3dopts_dict.get('depth')
        )
        
        return init_opts,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3dopts
    def draw_path(self,path_data,x_axis3d_opts,y_axis3d_opts,z_axis3d_opts,grid3d_opts,init_opts):
        path=( 
            Line3D(init_opts=init_opts)
                .add(
                series_name="",
                shading="color",
                data=path_data,
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
                )
            )
        return path
    def Threaten_rate(self,p:Loc):
        #根据所给的坐标点返回威胁概率值，需要不同环境模型自定义
        #特殊情况：地图边界
        if p.x<0 or p.x>self.width or p.y <0 or p.y>self.width or p.z <0 or p.z>self.h:
            return 1
        for th in self.buildings:
            if th.check_threaten(p)>0:
                return 1
        return 0
    def Draw_threaten_map(self,h=0):
        #创建完整威胁地图(二维)
        map=np.zeros((self.width,self.len,1))
        for i in range(self.width):
            for j in range(self.len):
                map[i,j,0]=self.Threaten_rate(Loc(i,j,h))
        return map
    def Reset_Thread_UAV(self,uav):
        uav.reset()   #随机重置状态


    #采用多线程优化uav的重置过程
    def Scene_Random_Reset(self):
        self.reset_threads=[]
        for indx,uav in enumerate(self.Agents):
            #普通调用
            uav.reset()   #随机重置状态

            
    def Set_Space_Scale(self,len:int,width:int,h:int):
        #重新设置空间大小
        self.len=len
        self.width=width
        self.h=h

    def Draw_static_map(self):
        pass

    def Check_uav_Done(self):
        #检查环境中的所有智能体是否都进入任务完成状态
        #输出参数：
        #       flag    ：  状态1：所有Agent都完成任务  0：存在Agent没有完成任务
        for agent in self.Agents:
            if not agent.done:
                return False
        return True
    
    def check_threaten(self,p:Loc):
        pass
    
    def check_total_threaten(self,x,y,z):
        try:
            if self.map[int(x),int(y)]>0:
                return 1
            else:
                return 0
        except Exception as e:
            print(e.args)  #输出异常信息
            return 1
    
    def run(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        pass

    def Choose_Action(self,index:int,eps=0.2):
        #epsilon贪心策略选取动作索引
        #输入参数
        #       index   ：  智能体所在Agents集合的索引
        #       eps     ：  eps贪心概率
        #输出参数
        #       action  ：  动作值
        if self.Is_AC==1:
            #actor-critic框架(包括SAC)
            sample = random.random()
            if  sample > eps:
                state = self.Agents[index].state()  #指定智能体的状态值
                tensor_state=FloatTensor(np.array(state))
                probs = self.Agents[index].Trainer.actor(tensor_state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                return torch.tensor([[action.item()]], device=device) 
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)   #随机选取动作
        elif self.Is_AC==2:  #DDPG算法
            sample = random.random()
            if  sample > eps:
                state = self.Agents[index].state()  #指定智能体的状态值
                tensor_state=FloatTensor(np.array(state))
                #state = torch.tensor([state], dtype=torch.float).to(self.device)
                action = self.Agents[index].Trainer.actor(tensor_state).item()
                # 给动作添加噪声，增加探索
                action = action + 0.01* np.random.randn(1)
            else:
                #随机选取动作
                return torch.tensor([random.uniform(-1,1)], device=device)   #随机选取动作
            return action
        elif self.Is_AC==3:  #SAC算法(离散动作空间)
            state = self.Agents[index].state()  #指定智能体的状态值
            tensor_state=FloatTensor(np.array(state))
            probs = self.Agents[index].Trainer.actor(tensor_state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return torch.tensor([[action.item()]], device=device) 
        elif self.Is_AC==4:  #SAC算法(连续动作空间)
            state = self.Agents[index].state()  #指定智能体的状态值
            state = torch.tensor([state], dtype=torch.float).to(device)
            action = self.Agents[index].Trainer.actor(state)[0]
            return torch.tensor([[action.item()]], device=device) 
        else:
            #DQN框架类型
            state=self.Agents[index].state()  #指定智能体的状态值
            tensor_state=FloatTensor(np.array(state))
            sample = random.random()
            if  sample > eps:
                with torch.no_grad():
                    y=self.Agents[index].Trainer.q_local(Variable(tensor_state).type(FloatTensor))
                    value=y.data.max(0)[1].view(1, 1) 
                    return  value   #根据Q值选择行为
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)   #随机选取动作
    
    #动作选择方法2，将决定动作的权力交给训练器
    def Choose_Action2(self,index:int,eps=0.2):
        #epsilon贪心策略选取动作索引
        #输入参数
        #       index   ：  智能体所在Agents集合的索引
        #       eps     ：  eps贪心概率
        #输出参数
        #       action  ：  动作值
        state = self.Agents[index].state()  #获取状态
        return self.Agents[index].Trainer.get_action(state,eps) #返回动作值
        

    def Evaluation_Action(self,index:int):
        #对指定智能体执行动作后的状态进行评价
        pass
    
    
    def record(self,result):
        #统计训练信息
        data=[]
        pass
    
    def Reset_Result(self,eps_rate):
        #重置统计量
        self.result={'success':0,'lose':0,'meet_threaten':0,'normal':0,'loss':0,'sum_epoch':0,'eps':eps_rate,'score':0,'average_score':0,'step':0}   #返回的推演信息：success-完成任务的智能体数目，failed：失败的智能体数目，meet_threaten-遇到威胁的次数
    
    #线程启用函数（离线策略）
    def run_thread_OffPolicy(self,uav,ind,eps_rate):
        if uav.done:
            return #该智能体已完成任务
        #智能体决策
        state_test=uav.state()                          #当前的状态图
        #action=self.Choose_Action(ind,eps_rate)   #epsilon贪心策略选取动作值
        action=self.Choose_Action2(ind,eps_rate)   #采用新版的动作选择方式
        next_state, reward, done, info= self.Move_Agent(ind,action)  #根据选取的动作改变状态，获取收益
        self.Run_statistics(info)   #对智能体运行信息进行统计
        #存储交互经验（存放tensor版的数据）
        uav.Trainer.Push_Replay(
            (FloatTensor(np.array([state_test])), 
            action, 
            FloatTensor([[reward]]), 
            FloatTensor(np.array([next_state])), 
            FloatTensor([[done]])))

        #将普通格式的数据存放到回放池中
        uav.Trainer.replay_memory.add(state_test, action, reward, next_state, done)
        if  len(uav.Trainer.replay_memory.buffer)>uav.Trainer.Batch_Size:
            b_s, b_a, b_r, b_ns, b_d,idx,weights = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            uav.transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d,'idx':idx,'weights':weights}
    
    #线程启用函数(在线策略)
    def run_thread_OnPolicy(self,uav,ind,eps_rate):
        if uav.done:
            return #该智能体已完成任务
        #智能体决策
        state=uav.state()                          #当前的状态图
        #action=self.Choose_Action(ind,eps_rate)   #epsilon贪心策略选取动作值
        action=self.Choose_Action2(ind,eps_rate)   #采用第二种动作选择方式
        next_state, reward, done, info= self.Move_Agent(ind,action)  #根据选取的动作改变状态，获取收益
        self.Run_statistics(info)   #对智能体运行信息进行统计
        # uav.transition_dict['states'].append(FloatTensor(np.array([state])))
        # uav.transition_dict['actions'].append(action)
        # uav.transition_dict['next_states'].append(FloatTensor(np.array([next_state])))
        # uav.transition_dict['rewards'].append(FloatTensor([[reward]]))
        # uav.transition_dict['dones'].append(FloatTensor([[done]]))
        #存放非tensor格式数据
        uav.transition_dict['states'].append(state)
        uav.transition_dict['actions'].append(action)
        uav.transition_dict['next_states'].append(next_state)
        uav.transition_dict['rewards'].append(reward)
        uav.transition_dict['dones'].append(done)
                    

    def run_eposide(self,eps_rate=0.1):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #返回参数
        #       result    ：  推演结果，dict类型
        self.Reset_Result(eps_rate)     #重置统计量
        self.Scene_Random_Reset()  #计算机随机生成场景
        self.threads=[]
        if self.Is_On_Policy==1:
            #在线策略训练，AC类算法训练方式
            for ind,uav in enumerate(self.Agents):
                uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}          
            while(1):
                self.run()          #场景元素进行变更
                for ind,uav in enumerate(self.Agents):
                    #多线程调用
                    thread = threading.Thread(target=self.run_thread_OnPolicy, args=(uav,ind,eps_rate,))  #在线训练
                    self.threads.append(thread) 
                    thread.start()
                # 等待所有线程完成
                for thread in self.threads:
                    thread.join()

                #如果所有智能体完成任务，推出当前推演
                if self.Check_uav_Done():
                    break
            #train_info=self.train_on_policy()   #在线训练,运行完一幕才开始训练
            train_info=self.update()   #在线策略,运行完一幕才开始训练
        else:  
             #离线训练策略
            while(1):
                self.run()          #场景元素进行变更
                for ind,uav in enumerate(self.Agents):
                    #多线程调用
                    uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} #重置
                    self.run_thread_OffPolicy(uav,ind,eps_rate)  #顺序运算
                    #thread = threading.Thread(target=self.run_thread_OffPolicy, args=(uav,ind,eps_rate,))
                    #self.threads.append(thread)
                    #thread.start()

                # 等待所有线程完成
                # for thread in self.threads:
                #     thread.join()
                    
                #train_info=self.train_off_policy()   #执行一个动作训练一次
                train_info=self.update()   #离线策略，执行一个动作训练一次
                #如果所有智能体完成任务，推出当前推演
                if self.Check_uav_Done():
                    break
        self.Train_statistics(train_info)  #统计训练结果，进行格式化保存
        self.epoch+=1   #运行了一个周期
            
        #记录这段训练数据
        if self.epoch%self.print_loop==0:
            for uav in self.Agents:
                uav.record_list()
        
        #联邦合并周期
        if self.epoch%self.FL_Loop==0 and self.Is_FL: # and self.epoch<50:
            if self.Is_AC:
                #AC版联邦聚合
                self.Federated_Learning_AC() 
            else:
                #self.Federated_Learning_choice() #选择性聚合
                self.Federated_Learning()   #进行联邦聚合
        data_result=self.generate_train_result()  #将这一轮的训练结果进行统计
        self.render(data_result)  #进行可视化操作，并将之存放入数据库中
        return self.result
    def generate_train_result(self):
        #生成训练的统计结果
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_score=self.Agents[0].total_score  #总分
        epoch=self.Agents[0].Train_epoch  #迭代次数
        train_time=time.time()-self.Agents[0].Train_start  #训练时长
        path_len=self.Agents[0].path_len   #当前航线长度
        start2goal=self.Agents[0].start2goal  #起点与终点的距离
        len_Astar=self.Agents[0].len_Astar   #Astar算法的航线长度
        ReachGoal=self.Agents[0].reach_goal  #是否到达目标点
        Train_info_line= {
            'uuid':  str(uuid.uuid4()),  
            'date': current_time,
            'html_dir': '<html lang="en"></html>',
            'train_score': train_score,
            'epoch': epoch,
            'Is_Scored': 0,
            'ReachGoal': 0,
            'train_time': train_time,
            'path_len': path_len,
            'score_quality': 0,
            'score_len': 0,
            'score_threaten': 0,
            'start2goal':start2goal,
            'len_Astar':len_Astar,
            'ReachGoal':ReachGoal,
        }
        return Train_info_line
    def Run_statistics(self,info):
        #对运行结果进行统计
        self.result[info]=self.result[info]+1             #统计运行结果

    def Train_statistics(self,train_info):
        #对训练结果进行统计
        num=len(train_info)
        for index,info in enumerate(train_info):
            if type(info['loss'])!=int:
                self.result['loss']+=info['loss'].item()
            else:
                self.result['loss']+=info['loss']
            self.result['sum_epoch']+=info['sum_epoch']
            self.result['score']+=self.Cal_Score()[0]
            self.result['average_score']+=self.Cal_Score()[1]
            self.result['step']+=self.Agents[index].Step

        #平均
        if num!=0:
            self.result['loss']/=num
            self.result['sum_epoch']/=num
            self.result['score']/=num
            self.result['average_score']/=num
            self.result['step']/=num

    def Cal_Score(self):
        #计算环境中所有智能体的总得分与平均得分
        #返回参数
        #   sum_score       :   总体得分
        #   average_score   :   平均得分
        sum_score=0
        average_score=0
        for agent in self.Agents:
            sum_score+=agent.score
        return sum_score,sum_score/len(self.Agents)
    
    def run_XML_scene(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #根据XML生成场景
        pass 
    
    
    def Load_Scene_FromXML(self,XML_path="../config/FMEC.xml"):
        #从xml文件中对环境成员进行配置，用于特定场景的强化学习训练或者训练效果测试
        #解析XML文件,从XML文件中解析成dict
        p_dict=XML2Dict(XML_path)
        if p_dict!=None:
            env_dict=p_dict.get('env')
            if env_dict!=None:
                #根据参数初始化Agent
                agents_params=env_dict.get('Agents')
                if agents_params != None:
                    agents=agents_params.get('Agent')
                    for agent_param in agents:
                        obj_agent=self.AgentFactory.Create_Agent(agent_param,self)
                        agent_trainer=agent_param.get('Trainer')
                        if obj_agent != None:
                            agent_trainer['name']=obj_agent.name   #同一名称
                            obj_agent.Trainer=self.TrainerFactory.Create_Trainer(agent_trainer)   #创建智能体专属的训练器
                            self.Agents.append(obj_agent)
                
                #根据参数初始化Threaten
                threaten_params=env_dict.get('buildings')
                if threaten_params != None:
                    buildings=threaten_params.get('Threaten')
                    for threaten_param in buildings:
                        obj_threaten=self.ThreatenFactory.Create_Threaten(threaten_param,self)
                        if obj_threaten != None:
                            self.buildings.append(obj_threaten)
        
                #根据参数初始化训练器
                trainer_params=env_dict.get('Trainer')
                if trainer_params != None:
                    self.Trainer=self.TrainerFactory.Create_Trainer(trainer_params)
    
    def train(self):
        #根据replay_buffer中的数据进行训练
        re=self.Trainer.learn()
        return re
    

    def Federated_Learning_AC(self):
        #AC版联邦聚合
        global_model=copy.deepcopy(self.Agents[0].Trainer.actor)   #复制模型
        for k in global_model.state_dict().keys():
            for ind,uav in enumerate(self.Agents):
                if ind==0:
                    continue
                global_model.state_dict()[k]+=uav.Trainer.actor.state_dict()[k]
            global_model.state_dict()[k]=torch.div(global_model.state_dict()[k], len(self.Agents))   #local平均  
        #为每个uav替换全局模型
        for uav in self.Agents:
            uav.Trainer.replace_param(global_model)


    def Federated_Learning(self):
        #对模型进行聚合
        #self.model_library=[]  #模型库
        # for uav in self.Agents:
        #     self.model_library.append(uav.SPN_param())  #保存所有UAV的参数
        # n=5   #聚合KL散度前5的模型
        #对每个UAV挑选目标SPN
        for uav in self.Agents:
            start=time.time()
            target_SPN=[]
            SPN_KL=[]
            if len(uav.Trainer.replay_memory.buffer)>uav.Trainer.Batch_Size:
                b_s, b_a, b_r, b_ns, b_d = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            else:
                b_s, b_a, b_r, b_ns, b_d = uav.Trainer.replay_memory.sample2(len(uav.Trainer.replay_memory.buffer)-1)
            test_sample = b_s #测试样本
            SelfPolicy=uav.Trainer.get_policy_DFRL(test_sample)  #自身UAV的策略分布
            for target_uav in self.Agents:
                target_policy=target_uav.Trainer.get_policy_DFRL(test_sample)
                #KL_=kl_divergence(target_policy,SelfPolicy)
                KL_=[kl_divergence_FDRL(tp,Sp) for tp,Sp in zip(target_policy,SelfPolicy)]
                KL_=sum(KL_)/len(KL_)    #采样样本的KL散度均值
                target_SPN.append(target_uav.Trainer.SPN_param())
                SPN_KL.append(KL_)
            #SPN_aggregation= heapq.nsmallest(uav.Top_G, SPN_KL) #计算出待聚合的SPN
            sorted_indices = sorted(enumerate(SPN_KL), key=lambda x: x[1], reverse=False)
            SPN_aggregation=[target_SPN[index] for index, value in sorted_indices[:uav.Top_G]]
            with torch.no_grad():
                aggregation_model=copy.deepcopy(uav.Trainer.SPN_param())   #复制自己的模型
                for k in aggregation_model.state_dict().keys():
                    for model in SPN_aggregation:
                        aggregation_model.state_dict()[k]+=model.state_dict()[k]
                    aggregation_model.state_dict()[k].data.copy_(torch.div(aggregation_model.state_dict()[k], len(SPN_aggregation)))   #local平均  
            end=time.time()
            #uav.Update_SPN(aggregation_model)  #直接更新SPN参数(FedAvg)
            uav.Train_time+=(start-end)  #统计服务端的运行时间
            uav.Update_SPN_Soft(aggregation_model)  #软更新SPN参数（双端联邦学习）


    #选择式联邦训练
    def Federated_Learning_choice(self):
        #对模型进行聚合
        mse_loss = torch.nn.MSELoss()   #损失函数
        with torch.no_grad():
            for ind_uav,uav_p in enumerate(self.Agents):
                global_model=copy.deepcopy(uav_p.Trainer.q_local)   #复制模型
                global_model_target=copy.deepcopy(uav_p.Trainer.q_target)   #复制模型
                FL_list=[]  #待聚合的UAV
                test_state=[]   #测试状态集合

                #构造测试状态集合
                transitions=uav_p.Trainer.replay_memory.sample(10)   #采样10个样本
                from collections import namedtuple
                Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
                batch = Transition(*zip(*transitions))
                states=torch.cat(batch.state)
                q_value=uav_p.Trainer.q_local(states)   #自己状态预估的Q值,作为选择聚合模型的标准


                #迭代所有无人机，选择评分类似的UAV
                for ind,uav in enumerate(self.Agents):
                    if ind==ind_uav:   #不聚合自己的模型
                        continue
                    q_u=uav.Trainer.q_local(states)   #用所给的状态估计Q值
                    loss = mse_loss(q_value, q_u)
                    FL_list.append([ind,loss])  # 添加编号信息和损失信息

                sorted_list = sorted(FL_list, key=lambda x: x[1])   #按损失大小进行排序
                chosed_list=sorted_list[:len(sorted_list) // 2]   #取loss前一半的模型进行聚合

                #进行模型聚合
                for k in global_model.state_dict().keys():
                    for it in chosed_list:
                        global_model.state_dict()[k]+=self.Agents[it[0]].Trainer.q_local.state_dict()[k]
                        global_model_target.state_dict()[k]+=self.Agents[it[0]].Trainer.q_target.state_dict()[k]
                    global_model.state_dict()[k].data.copy_(torch.div(global_model.state_dict()[k], len(chosed_list)+1))   #local平均  
                    global_model_target.state_dict()[k].data.copy_(torch.div(global_model_target.state_dict()[k], len(chosed_list)+1))   #target平均  


                uav_p.Trainer.replace_param(global_model)
                #uav_p.Trainer.replace_target_param(global_model)

    #附带联邦学习的训练
    def train_off_policy_FL(self):
        #离线训练（有经验回放池）
        re=[]
        for uav in self.Agents:
            item=uav.Trainer.learn_off_policy()
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step
            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            for ue in uav.UEs:
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/len(uav.UEs)   #平均等待时间
            uav.infos.append(item) 
            re.append(item)
        
        #联邦学习聚合
        if self.Is_FL:
            #self.Federated_Learning()
            self.Federated_Learning_choice()
        return re

    def train_off_policy(self):
        #离线训练（有经验回放池）
        re=[]
        for uav in self.Agents:
            #串行计算
            item=uav.Trainer.learn_off_policy()
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            item['ALL_UEs_F']=0
            item['ALL_UEs_D']=0
            for ue in uav.UEs:
                item['ALL_UEs_D']+=ue.D_t 
                item['ALL_UEs_F']+=ue.F_t 
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/len(uav.UEs)   #平均等待时间
            #uav.infos.append(item) 
            re.append(item)
        return re
    
    def train_on_policy(self):
        #在线训练（没有经验回放池）
        re=[]
        for uav in self.Agents:
            item=uav.Trainer.learn_on_policy(uav.transition_dict)
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            for ue in uav.UEs:
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/len(uav.UEs)   #平均等待时间
            #uav.infos.append(item) 
            re.append(item)
        return re
    
    #通用型更新方法，适用于离线和在线策略
    def update(self):
        #离线训练（有经验回放池）
        re=[]
        for uav in self.Agents:
            #串行计算
            item=uav.Train_nn() #各自训练神经网络
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            for ue in uav.UEs:
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/(len(uav.UEs)+1)   #平均等待时间
            #uav.infos.append(item) 
            re.append(item)
        return re
    
    
    def store_experience(self,replay):
        #存放经验信息
        self.Trainer.Push_Replay(replay)
