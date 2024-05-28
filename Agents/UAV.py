import time
from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
from PathPlan.RRT import *
from GN import *
import random
import csv
import copy
import  os
root=os.path.dirname(os.path.abspath(__file__)) #当前根目录
from  torch.autograd import Variable 
class UAV(BaseAgent):
    def __init__(self,param:dict,env=None) -> None:
        #初始化
        #输入参数：
        #       param   :   初始化参数字典
        super().__init__(param)   #初始化基类
        #自定义附加参数添加在如下地方：
        self.param=param  #保存初始化参数信息
        #UAV特有属性
        self.name=param.get("name")  #无人机名称
        self.j=param.get("j")  #无人机编号
        self.V_vector=Loc(0,0,0)  #速度矢量
        self.Max_V=int(param.get("Max_V"))   #最大速度
        self.Steering_angle=float(param.get("Steering_angle"))/180*math.pi   #最大转向角
        self.V=self.Calc_V()     #计算速度大小
        #原实验设置
        self.R=int(param.get("R"))+0.5*self.j   #计算范围
        self.R_comm=int(param.get("R_comm"))+0.75*self.j   #通讯范围
        self.ac=int(param.get("Acceration"))+0.25*self.j   #加速度大小
        self.Max_Step=int(param.get("Max_Step"))   #最大步长
        self.Step=0
        self.f_max=float(param.get("f_max"))+0.1*self.j   #时钟频率

        #UAV动作空间
        self.act=[
            [self.ac,0],[self.ac,0.25*math.pi],[self.ac,0.5*math.pi],[self.ac,0.75*math.pi],[self.ac,math.pi],[self.ac,1.25*math.pi],[self.ac,1.5*math.pi],[self.ac,1.75*math.pi] ,
            [0,0],
            [0.5*self.ac,0],[0.5*self.ac,0.25*math.pi],[0.5*self.ac,0.5*math.pi],[0.5*self.ac,0.75*math.pi],[0.5*self.ac,math.pi],[0.5*self.ac,1.25*math.pi],[0.5*self.ac,1.5*math.pi],[0.5*self.ac,1.75*math.pi]      
                ]             
        self.path=[]  #路径点
        self.V_record=[]  #记录速度大小
        self.R_record=[]  #记录奖励大小
        self.Trainer=None
        #读取能耗参数
        Power_param=param.get("Power_param")
        #飞行能耗参数
        Fly_power=Power_param.get("Fly_power")
        self.P_i=float(Fly_power.get("P_i"))
        self.v_0=float(Fly_power.get("v_0"))
        self.d_0=float(Fly_power.get("d_0"))
        self.rho=float(Fly_power.get("rho"))
        self.s=float(Fly_power.get("s"))
        self.A=float(Fly_power.get("A"))+0.03*self.j
        self.P_b=float(Fly_power.get("P_b"))
        self.F_b=float(Fly_power.get("F_b"))
        self.xi=0.8+0.02*self.j   #异构参数
        self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
        #通讯能耗
        self.Communication_power=float(Power_param.get("Communication_power"))
        #UEs配置参数
        self.UEs_param=param.get("UEs_param")
        #样本集合
        self.transition_dict= {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        self.env=env  #设置所处的环境
        #构建状态图层
        self.Min_dis=self.R  #最小距离障碍物的距离
        #训练数据记录
        self.CsvWriter=None
        self.infos=[]   #训练的反馈列表
        self.Init_Record_Mod()
        #目标点
        self.goal=Loc(75,75,0)
        #数据收集情况
        self.data_num=0

        #无人机执行UE的任务卸载
        #构造UEs
        self.UEs=[]
        self.Covered_UEs=[]
        self.comm_UEs=[]
        self.num_covered_ue=0    #覆盖的可收集ue数目
        self.num_comm_ue=0    # 通讯范围中的可收集ue数目
        self.num_UEs=int(self.UEs_param.get('UES_num'))
        for i in range(self.num_UEs):
            param_ue=copy.copy(self.UEs_param)
            param_ue['uav']=self
            param_ue['position']=Loc(random.uniform(0,self.env.len),random.uniform(0,self.env.width),0)   #设置坐标
            self.UEs.append(self.env.AgentFactory.Create_Agent(param_ue))

        self.Closed_UEs=[]   #附近可用的目标ue
        self.energy_cost_total=0   #总消耗能量
        self.task_collect=0  #总收集任务量
        self.task_executed=0  #处理的任务总量
        self.energy_efficient=0  #能效指标
        self.cover_rate=0        #覆盖率指标
        self.ue_waiting_time=0   #累计等待时间指标
        self.sum_KL=[]     #与其他UAV决策模型的KL散度之和，每决策一次就在原基础上增加

        #DSN训练器       
        self.DSNs_param=param.get("DSNs_param")   #DSNs配置参数
        self.step_skill_max=int(self.DSNs_param.get("step_skill_max"))   #单个技能的最大原子动作序列长度
        self.DSN_num=int(self.DSNs_param.get("DSN_num"))
        self.DSN_gamma=float(self.DSNs_param.get("DSN_gamma"))  #技能奖励折扣
        self.DSN_Trainer=[] 
        self.DSNs=[] 
        #self.Load_DSN()  #载入DSN训练器
        #DFRL训练所需参数
        self.DFRL_param=param.get("DFRL_param")   #DFRL配置参数
        self.update_rate=float(self.DFRL_param.get("update_rate"))   #更新率配置参数
        self.Top_G=int(self.DFRL_param.get("Top_G"))   #聚合前G个模型参数
        #技能4所需状态
        self.area_time=np.zeros((1,1,1,25))  #统计在九宫格每个区域的停留时间
        self.area_task=np.zeros((1,1,1,25))  #统计在九宫格每个区域采集任务数目的时间

        #最大覆盖贪心策略
        self.route=[[250,150],[150,150],[150,450],[50,450],[50,50],[450,50],[450,450],[250,450],[250,350],[350,350],[350,150],[250,250]]  #巡逻环路
        self.goal_ind=0
        #其他测试的SPN
        self.SPNs=[]
        #self.Load_SPN()
        #训练时间与测试时间统计
        self.Train_time=0
        self.Testing_time=0
        #每个时隙采用的函数名
        self.update_function_name=param.get("update_function_name")
        self.update_function= getattr(self,self.update_function_name)  
        #状态空间生成的函数
        self.state_function_name=param.get("state_function_name")
        self.state_function= getattr(self,self.state_function_name)
        #用于目标点搜索的参数
        self.goal=Loc(470,470,0)
        #数据收集情况
        self.data_num=0
        #Astar算法模块
        self.sub_granularity=int(param.get("sub_granularity"))# 子任务粒度
        self.map_granularity=int(param.get("map_granularity")) # 任务栅格粒度
        #self.granularity=1 # 粒度调整
        self.Planner=RRTPlanner(self.env)  #RRT算法
        self.sub_goals=[]  #拆分的子目标集合
        self.APF_Enabled=int(param.get("APF_Enabled"))  #是否支持APF方法
        #生成子任务
        _,path=self.Cal_SubTask(self.goal)
        #训练统计数据，存放到数据库中
        self.total_score=0   #整个任务的得分
        self.Train_start=time.time()  #训练开始的时间
        self.Train_pointcut=time.time()  #统计切点的时间
        self.path_len=0     #当前航线的长度
        self.Train_epoch=0   #当前迭代次数
        self.reach_goal=0  #是否到达目标点
        self.start2goal=Eu_Loc_distance(self.position,self.sub_goals[-1])  #记录目标点与起点的距离
        self.len_Astar=calculate_path_len(path)    #调用Astar算法得到的航线长度。

    #根据subgoal与周边障碍物的情况微调subgoal位置(斥力与引力模型)
    def Adjust_subgoal(self):
        new_sub_goals=[]
        subgoal_mindis=[]   #子目标点与威胁的最小距离
        for subgoal in self.sub_goals:
            total_force=Loc(0,0,0)
            total_force=self.cal_force(subgoal)  #计算合力
            new_goal=subgoal+total_force
            new_sub_goals.append(new_goal)
            subgoal_mindis.append(self.cal_minDis2threaten(new_goal))   #加入与障碍物距离最小的距离
        self.sub_goals=new_sub_goals
        return False
    def cal_minDis2threaten(self,new_goal):
        #计算新的子目标点与所有障碍物的最小距离
        min_dis=9999
        for threaten in self.env.buildings:
            min_dis=min(min_dis,Eu_Loc_distance(new_goal,threaten.position)-threaten._R)
        return min_dis

    def cal_force(self,subgoal):
        #计算某个子目标与威胁之间的斥力(只考虑阈值范围内的威胁)
        cum_force=0 #累积受力，过大会触发重新规划
        f=Loc(0,0,0)  #初始化合力
        total_force=Loc(0,0,0)
        for threaten in self.env.buildings:
            if threaten.v.x==0 and threaten.v.y==0 and threaten.v.z==0:
                #威胁没有速度
                continue
            dis=Eu_Loc_distance(subgoal,threaten.position)  #与威胁圆心的直线距离
            
            dis2edge=dis-threaten._R   #距离威胁边缘的半径
            if dis2edge>60:
                continue  #距离过远，不考虑作用力
            w=1  #引力参数
            v=Eu_Loc_distance(Loc(0,0,0),threaten.v)  #计算威胁的速度大小
            f_threaten1=min(1,w*threaten._R/(dis2edge*dis2edge))   #引力or斥力大小
            f_threaten1_seta=calculate_angle(subgoal,threaten.position)   #引力方向，记住目标点->威胁中心的方向角
            f_threaten1_vec=Loc(0,0,0)
            v_threaten_seta=calculate_angle(Loc(0,0,0),threaten.v)   #计算威胁速度的方位角

            #只考虑斥力
            if dis2edge<0:
                f_threaten1=max(-dis2edge,2)  #在威胁内部，另外考虑斥力大小
            f_threaten1_vec=Loc(-f_threaten1*math.cos(f_threaten1_seta),-f_threaten1*math.sin(f_threaten1_seta),0)
            
            f_threaten2=min(1,v*threaten._R/(dis2edge*dis2edge))  #运动力大小
            f_threaten2_vec=Loc(f_threaten2*math.cos(v_threaten_seta),f_threaten2*math.sin(v_threaten_seta),0)   #运动力的向量
            cum_force+=(f_threaten1+f_threaten2)  #加上运动力和引力斥力
            total_force=total_force+f_threaten1_vec+f_threaten2_vec  #力的合并

            if cum_force>100:
                #累积力过大，应当重新规划
                self.Cal_SubTask_Dynamic()
                return True
            
        return total_force

    def Cal_SubTask_Dynamic(self,goal,Loc):
        #动态预测环境下的子任务划分
        pass

    def Cal_SubTask(self,goal:Loc):
        self.Planner.Set_StepSize(self.sub_granularity)  #设置Astar算法参数
        Path,path_ori=self.Planner.getPath(self.position,goal)  #得到栅格路径，还需要将之转换为真实路径
        if Path==None:
            #没有得到路径
            return None,None
        #self.sub_goals=[self.position] #重置子任务集合
        self.sub_goals=[]
        for cp in Path:
            self.sub_goals.append(cp)
        return self.sub_goals,path_ori
    #载入测试用的SPN
    def Load_SPN(self):
        directory_path = root+"/.."+'/Mod/'  #存放DSN模型的路径
        file_list = os.listdir(directory_path)
        pth_files = [file for file in file_list if file.endswith(".pth") and file.startswith("actor")]
        pth_files=sorted(pth_files, key=lambda x: int(x.split('.')[0].split('_')[3]))  #按文件名进行排序
        param={'w':87,'hiden_dim':128,'output':9}
        for dict_file in pth_files:
            model=PolicyNet_SAC(param)
            state_dict=torch.load(directory_path+'/'+dict_file)
            model.load_state_dict(state_dict['model'])
            self.SPNs.append(copy.copy(model))

    def Calc_Fly_Power(self):
        #计算飞行功率
        self.V=self.Calc_V() 
        induced=self.P_i*math.sqrt(math.sqrt(1+(self.V**4)/(4*(self.v_0**4)))-(self.V**2)/(2*(self.v_0**2)))
        parasite=0.5*self.d_0*self.rho*self.s*self.A*(self.V**3)
        blade=self.xi*self.P_b*(1+3*(self.V**2)/(self.F_b**2))
        return induced+parasite+blade 
    def Calc_V(self):
        #根据速度向量计算速度大小
        V=Eu_Loc_distance(Loc(0,0,0),self.V_vector)
        if V>self.Max_V:  #自动调整速度
            self.V_vector.x=self.V_vector.x*(self.Max_V/V)
            self.V_vector.y=self.V_vector.y*(self.Max_V/V)
            V=self.Max_V
        return V
    def Set_Env(self,env):
        #设置所属环境类
        self.env=env
    
    #载入DSN
    def Load_DSN(self):
        #只载入模型
        self.DSNs=[] 
        DSN_path=root+"/.."+'/Mod/DSNs/DSN_'  #存放DSN模型的路径
        for i in range(self.DSN_num): 
            path=DSN_path+str(i+1)+'.pth'
            mod=torch.load(path).to(device)
            self.DSNs.append(mod)
  
    #初始化记录模块
    def Init_Record_Mod(self):
        #初始化训练记录模块
        import datetime
        cur_time=datetime.datetime.now()
        cur_time= cur_time.strftime('%m_%d_%Y(%H_%M_%S)')
        path='logs/%s_%s.csv' %(self.name,cur_time)
        file=open(path,'a+',newline="")
        self.CsvWriter=csv.writer(file)
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','ALL_UEs_D','ALL_UEs_F','energy_cost','task_collect','Energy_Efficent','UE_waiting_time','Covered_rate','task_executed','executed_rate','KL','Train_time','Testing_time'])

    #记录训练数据
    def record_list(self):
        ALL_UEs_D=ALL_UEs_F=executed_rate=task_executed=Covered_rate=sum_epoch=score=average_score=eps=success=lose=meet_threaten=loss=step=energy_cost=task_collect=UE_waiting_time=Energy_Efficent=0
        list_len=len(self.infos)
        avg_KL=[]
        
        sum_epoch=self.Trainer.epoch
        score=self.score
        average_score=self.score
        loss+=0   #loss的绝对值          
        #无人机采集UE任务所需数据
        for ue in self.UEs:
            ALL_UEs_D+=ue.D_t 
            ALL_UEs_F+=ue.F_t 
            UE_waiting_time=ue.Caculate_wait_time  #统计每一个ue的等待时间

        energy_cost=self.energy_cost_total
        task_collect=self.task_collect
        Energy_Efficent=self.task_collect/(self.energy_cost_total+0.001)
        Covered_rate=len([UE for UE in self.UEs  if UE.Is_covered==1])/(len(self.UEs)+1)
        task_executed=self.task_executed
        executed_rate=(task_executed/(task_collect+1))
        if len(avg_KL)==0:
            avg_KL=[x /self.Max_Step for x in self.sum_KL]    #计算平均KL散度
        else:
            avg_KL=list(map(lambda x, y: x + y, self.sum_KL, [x /self.Max_Step for x in self.sum_KL] ))
        self.CsvWriter.writerow([sum_epoch,self.Trainer.epoch,score,average_score,eps,success,lose,meet_threaten,loss,ALL_UEs_D,ALL_UEs_F,energy_cost,task_collect,Energy_Efficent,UE_waiting_time,Covered_rate,task_executed,executed_rate,[x  for x in avg_KL],self.Train_time,self.Testing_time])
            #self.CsvWriter.flush()
        self.infos=[]  #清空列表

    def update(self,action):
        #根据反射机制返回目标动作
        return self.update_function(action) #根据反射执行动作
    
    def state(self):
        #根据模块调用对应的状态空间构建向量
        # function = getattr(self,self.state_function_name)  
        # return function() #根据反射执行动作
        return self.state_function()
    
    def Train_nn(self):
        start=time.time()
        #训练神经网络
        re=self.Trainer.update(self.transition_dict)
        end=time.time()
        self.Train_time+=(end-start)  #统计训练神经网络的时间
        return re

    # def reset(self):
    #     self.Step=0
    #     self.score=0
    #     self.done=False
    #     self.path=[] 
    #     self.V_record=[]  #记录速度大小
    #     self.R_record=[]  #记录奖励大小
    #     self.position=Loc(int(self.param.get("position").get('x')),int(self.param.get("position").get('y')),int(self.param.get("position").get('z')))  #初始化在空间中的位置
    #     self.V_vector=Loc(0,0,0)  #速度矢量
    #     self.V=self.Calc_V()     #计算速度大小
    #     self.infos=[]   #训练的反馈列表
    #     #数据收集情况
    #     self.data_num=0
    #     #目标点
    #     self.goal=Loc(75,65,0)
    #     seta=random.uniform(0,2*math.pi)
    #     self.goal=Loc(self.position.x+30*math.cos(seta),self.position.y+30*math.sin(seta),0)  #随机生成距离30m的终点
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     #构造UEs
    #     self.UEs=[]
    #     self.Covered_UEs=[]
    #     self.comm_UEs=[]
    #     self.num_covered_ue=0    #覆盖的可收集ue数目
    #     self.num_comm_ue=0    # 通讯范围中的可收集ue数目
    #     self.num_UEs=int(self.UEs_param.get('UES_num'))
    #     for i in range(self.num_UEs):
    #         param_ue=copy.copy(self.UEs_param)
    #         param_ue['uav']=self
    #         param_ue['position']=Loc(random.uniform(0,self.env.len),random.uniform(0,self.env.width),0)   #设置坐标
    #         self.UEs.append(self.env.AgentFactory.Create_Agent(param_ue))

    #     self.Closed_UEs=[]   #附近可用的目标ue
    #     self.energy_cost_total=0   #总消耗能量
    #     self.task_collect=0  #总收集任务量
    #     self.task_executed=0  #处理的任务总量
    #     self.energy_efficient=0  #能效指标
    #     self.cover_rate=0        #覆盖率指标
    #     self.ue_waiting_time=0   #累计等待时间指标
    #     self.sum_KL=[]        #累积KL散度值
    #     #技能4所需状态
    #     self.area_time=np.zeros((1,1,1,25))  #统计在九宫格每个区域的停留时间
    #     self.area_task=np.zeros((1,1,1,25))  #统计在九宫格每个区域采集任务数目的时间
    #     #最大覆盖贪心策略
    #     self.goal_ind=0
    #     #训练时间与测试时间统计
    #     # self.Train_time=0
    #     # self.Testing_time=0
    def reset(self,option=None):
        if option=="local reset":
            #局部重置
            self.Step=0
            self.score=0
            self.V=self.Calc_V()     #计算速度大小
        else:
            #全局重置
            self.Step=0
            self.score=0
            self.done=False
            self.path=[] 
            self.V_record=[]  #记录速度大小
            self.R_record=[]  #记录奖励大小
            #根据配置文件生成起点坐标
            #self.position=Loc(int(self.param.get("position").get('x')),int(self.param.get("position").get('y')),int(self.param.get("position").get('z')))  #初始化在空间中的位置
            self.V_vector=Loc(5,3,0)  #速度矢量
            seta=random.uniform(0,2*math.pi)   #随机初始化速度方向
            self.V_dir=seta   #速度方向
            self.V_vector.x=self.Max_V*math.cos(seta)
            self.V_vector.y=self.Max_V*math.sin(seta)  #初始速度大小为Min_V
            self.V=self.Calc_V()     #计算速度大小
            self.infos=[]   #训练的反馈列表
            #数据收集情况
            self.data_num=0
            #起点重置
            # x=random.uniform(10,490)
            # y=random.uniform(1,10)
            # self.position=Loc(x,y,0)
            self.position=Loc(20,20,10)   #固定起点
            #目标点
            #seta=random.uniform(0,2*math.pi)
            x=random.uniform(10,490)
            y=random.uniform(450,490)
            #self.goal=Loc(x,y,0)
            self.goal=Loc(480,480,25)  #固定目标点
            _,Astar_path=self.Cal_SubTask(self.goal)  #子任务拆分
            #记录子任务路径
            # with open('path_subtask.csv', 'w', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     for data in self.sub_goals:  # 假设有10次迭代
            #         csvwriter.writerow([data.x,data.y,data.z])
            #训练统计数据，存放到数据库中
            self.total_score=0   #整个任务的得分
            # self.Train_start=time.time()  #训练开始的时间
            # self.Train_pointcut=time.time()  #统计切点的时间
            self.path_len=0     #当前航线的长度
            # self.Train_epoch=0   #当前迭代次数
            self.reach_goal=0  #是否到达目标点
            self.start2goal=Eu_Loc_distance(self.position,self.goal)  #记录目标点与起点的距离
            self.len_Astar=calculate_path_len(Astar_path)    #调用Astar算法得到的航线长度。
    #选取动作
    def get_action(self,state,eps):
        start=time.time()
        action=self.Trainer.get_action(state,eps)
        end=time.time()
        self.Testing_time+=(end-start)  #统计测试时间
        return action
    def is_in_building(self,position):
        return False
    def Set_Max_Step(self,n:int):
        #设置任务的最大步长
        self.max_step=n

    def Set_State_Map(self,L:int,W:int,H:int):
        #设置状态图层形状
        #L图层长度，W图层宽度，H图层厚度
        self.state_map=np.zeros((1,H,L,W))

    def Set_Actions(self,Action_Dic):
        #通过设定好的动作集合初始化智能体的动作空间
        #Action_Dic为动作空间字典，包含了对应动作编号下的坐标（或姿态）变更规则
        #例如actions={'L':Loc(0,-1,0),'R':Loc(0,1,0),'U':Loc(1,0,0),'L':Loc(-1,0,0)}
        self.Actions=Action_Dic
    
    def Observation(self,env):
        #该函数实现环境状态感知，使智能体能感知到环境类中的部分或全部成员
        pass
    #直接更新SPN参数
    def Update_SPN(self,aggregation_model):
        self.Trainer.Replace_Model(aggregation_model)  #直接替换模型
    #双端联邦更新
    def Update_SPN_Soft(self,aggregation_model):
        #统计时间
        start=time.time()
        #先进行软更新
        self.Trainer.DFRL_sotf(aggregation_model,self.update_rate)   #DRFL算法软更新参数
        #进行自适应训练,只进行1次
        if len(self.Trainer.replay_memory.buffer)>self.Trainer.Batch_Size:
            b_s, b_a, b_r, b_ns, b_d = self.Trainer.replay_memory.sample2(self.Trainer.Batch_Size)
        else:
            b_s, b_a, b_r, b_ns, b_d = self.Trainer.replay_memory.sample2(10)
        sample = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}  #测试样本
        states = torch.tensor(sample['states'],dtype=torch.float).to(device)
        actions = torch.tensor(sample['actions'],dtype=torch.float).view(-1, 1).to(device)
        rewards = torch.tensor(sample['rewards'], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(sample['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(sample['dones'],dtype=torch.float).view(-1, 1).to(device)
        # 更新两个Q网络
        td_target = self.Trainer.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.Trainer.critic_1(states).gather(1, actions.long())
        critic_2_q_values = self.Trainer.critic_2(states).gather(1, actions.long())
        # 更新策略网络
        probs = aggregation_model(states)
        log_probs = torch.log(probs + 1e-8)
        #更新actor网络
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.Trainer.critic_1(states)
        q2_value = self.Trainer.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                            dim=1,
                            keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.Trainer.log_alpha.exp().to(device) * entropy - min_qvalue)
        self.Trainer.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Trainer.actor_optimizer.step()
        end=time.time()
        self.Train_time+=(end-start)


    # ██████╗  █████╗ ███╗   ██╗██████╗  ██████╗ ███╗   ███╗
    # ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔═══██╗████╗ ████║
    # ██████╔╝███████║██╔██╗ ██║██║  ██║██║   ██║██╔████╔██║
    # ██╔══██╗██╔══██║██║╚██╗██║██║  ██║██║   ██║██║╚██╔╝██║
    # ██║  ██║██║  ██║██║ ╚████║██████╔╝╚██████╔╝██║ ╚═╝ ██║
    # ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝     ╚═╝
    #随机选择动作
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     action=random.randint(0,10)   #随机选取动作
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append([self.position.x,self.position.y,self.V])
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=5
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=5
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=5
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=5
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     global_r+=(-0.005)*sum_power   #能量消耗惩罚
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     #覆盖惩罚
    #     sum_d=len([UE for UE in self.UEs if UE.Is_covered==0])   #统计没有被覆盖到的UEs数目
    #     global_r-=((sum_d/50)**2)  #未被覆盖的UE越多，惩罚越大
    #     #最大化cpu利用率
    #     global_r-=((self.f_max-cpu_rate))**2/5
    #     #UE待处理任务的惩罚
    #     # sum_F=len([UE.F_t for UE in self.UEs])   #统计所有UEs的待处理任务
    #     # global_r-=((sum_F/50)**2)  #未被覆盖的UE越多，惩罚越大
    #     global_r=global_r*5

    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         with open('path.csv', 'w', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile)
    #             for data in self.path:  # 假设有10次迭代
    #                 csvwriter.writerow(data)
    #         return global_r,True,'success'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 
                                                        
    #  ██████╗ ██████╗ ███████╗███████╗██████╗ ██╗   ██╗
    # ██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔══██╗╚██╗ ██╔╝
    # ██║  ███╗██████╔╝█████╗  █████╗  ██║  ██║ ╚████╔╝ 
    # ██║   ██║██╔══██╗██╔══╝  ██╔══╝  ██║  ██║  ╚██╔╝  
    # ╚██████╔╝██║  ██║███████╗███████╗██████╔╝   ██║   
    #  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝    ╚═╝                                                 
    #最大覆盖贪心策略
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计

    #     min_dis=99999
    #     min_act=0
    #     for tri_ind in range(9):
    #         act=self.act[tri_ind]  
    #         dx=self.V_vector.x+act[0]*math.cos(act[1])
    #         dy=self.V_vector.y+act[0]*math.sin(act[1])
    #         #判断相对位移
    #         px=self.position.x+dx   #变更x坐标
    #         py=self.position.y+dy   #变更y坐标标
    #         if Eu_Loc_distance(Loc(px,py,0),Loc(self.route[self.goal_ind][0],self.route[self.goal_ind][1],0))<min_dis:
    #             min_dis=Eu_Loc_distance(Loc(px,py,0),Loc(self.route[self.goal_ind][0],self.route[self.goal_ind][1],0))
    #             min_act=tri_ind
    #     act=self.act[min_act]  
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标标
    #     self.path.append([self.position.x,self.position.y,self.V])
    #     self.V_record.append(self.V)

    #     #如果到达既定目标点，变更目标点
    #     if Eu_Loc_distance(Loc(self.position.x,self.position.y,0),Loc(self.route[self.goal_ind][0],self.route[self.goal_ind][1],0))<=15:
    #         self.goal_ind=(self.goal_ind+1)%(len(self.route))
            
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     global_r+=(-0.01)*sum_power   #能量消耗惩罚
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理 
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目
    #             calcuated_task=calcuated_task/ue.F*ue.D    #将周期数转换成KB大小
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             global_r+=(0.01*calcuated_task*(1+0.00001*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 #global_r+=0.2   #额外奖励
    #                 ue.reset()
    #     #统计所有的UEs(简易版)
    #     sum_d=len([UE for UE in self.UEs if UE.Is_covered==0])   #统计没有被覆盖到的UEs数目
    #     global_r-=((sum_d/50)**2)  #未被覆盖的UE越多，惩罚越大

    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         with open('path.csv', 'w', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile)
    #             for data in self.path:  # 假设有10次迭代
    #                 csvwriter.writerow(data)
    #         return global_r,True,'success'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal'                                                 

    # ███████╗███╗   ███╗███████╗ ██████╗
    # ██╔════╝████╗ ████║██╔════╝██╔════╝
    # █████╗  ██╔████╔██║█████╗  ██║     
    # ██╔══╝  ██║╚██╔╝██║██╔══╝  ██║     
    # ██║     ██║ ╚═╝ ██║███████╗╚██████╗
    # ╚═╝     ╚═╝     ╚═╝╚══════╝ ╚═════╝
    #不使用分层强化学习
    def update_SAC(self,action):
        start_SAC=time.time()   
        global_r=0  #全局任务奖励
        self.Step+=1    #全局步长统计
        act=self.act[action]  #获取动作，得到加速度的大小与方向
        #速度变更
        self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
        self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
        self.V=self.Calc_V()   #计算速度大小,并自适应调整
        self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
        sum_power=self.P_fly+self.Communication_power   #计算总功率
        #判断相对位移
        self.position.x+=self.V_vector.x   #变更x坐标
        self.position.y+=self.V_vector.y    #变更y坐标
        self.path.append([self.position.x,self.position.y,self.V])
        self.V_record.append(self.V)
        #更新UE任务卸载部分的状态
        if self.position.x>=self.env.len:
            self.V_vector.x=(-self.V_vector.x)  #速度反转
            self.position.x+=self.V_vector.x   #变更x坐标
            global_r-=1
        if self.position.x<0:
            self.V_vector.x=(-self.V_vector.x)  #速度反转
            self.position.x+=self.V_vector.x   #变更x坐标
            global_r-=1
        if self.position.y>=self.env.width:
            self.V_vector.y=(-self.V_vector.y)  #速度反转
            self.position.y+=self.V_vector.y    #变更y坐标
            global_r-=1
        if self.position.y<0:
            self.V_vector.y=(-self.V_vector.y)  #速度反转
            self.position.y+=self.V_vector.y    #变更y坐标
            global_r-=1
        #UE运行
        for ue in self.UEs:
            ue.run()
        #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
        self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
        self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
        self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
        self.num_comm_ue=len(self.comm_UEs)
        #更新状态
        #global_r+=(-0.01)*sum_power   #能量消耗惩罚
        self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
        #统计25宫格的区域信息
        old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
        ind_x=int(self.position.x/100)   #x区域号
        ind_y=int(self.position.y/100)   #y区域号
        ind=ind_y*5+ind_x
        self.area_time[0,0,0,ind]+=1     #增加停留时间
        #无人机采集数据
        cpu_rate=0   #cpu利用率
        for ue in self.Covered_UEs:
            if ue.Now_covered==1:  #判断可收集UE
                #收集任务
                task=ue.trans()  #上传的任务量
                self.task_collect+=task     #收集的任务数目              
                #计算处理
                executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
                calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
                cpu_rate+=calcuated_task              #统计cpu利用率
                calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
                self.task_executed+=calcuated_task     #统计处理过的任务数目
                self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目
                #global_r+=(calcuated_task/sum_power)/10       #奖励为能效值
                #global_r+=(0.01*calcuated_task*(1+0*self.task_executed))
                #所有累积任务都处理完毕,获得额外奖励
                if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
                    ue.reset()
        #覆盖惩罚
        sum_d=len([UE for UE in self.UEs if UE.Is_covered==0])   #统计没有被覆盖到的UEs数目
        global_r-=((sum_d/100)**2)  #未被覆盖的UE越多，惩罚越大
        #最大化cpu利用率
        #global_r-=((self.f_max-cpu_rate)*sum_power/100)**2/5
        #UE待处理任务的惩罚
        # sum_F=len([UE.F_t for UE in self.UEs])   #统计所有UEs的待处理任务
        # global_r-=((sum_F/100)**2)  #未被覆盖的UE越多，惩罚越大
        #global_r=global_r*5
        end_SAC=time.time()
        time_SAC=end_SAC-start_SAC
        self.R_record.append(global_r)  #记录奖励大小
        if self.Step>=self.Max_Step:
            self.done=True
            self.score+=global_r
            # with open('path.csv', 'w', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     for data in self.path:  # 假设有10次迭代
            #         csvwriter.writerow(data)
            return global_r,True,'success'
        else:
            self.score+=global_r
            return global_r,False,'normal' 

    # # #简化版状态空间
    def state_SAC(self):
        state_start=time.time()
        #所有技能所需要的状态空间
        self.Min_dis=self.R
        state_map=np.zeros((1,1,1,87))
        state_map[0,0,0,0]=self.Step/150
        state_map[0,0,0,1]=self.position.x/500
        state_map[0,0,0,2]=self.position.y/500
        state_map[0,0,0,3]=self.V_vector.x/20
        state_map[0,0,0,4]=self.V_vector.y/20
        state_map[0,0,0,5]=self.task_executed/10000
        #停留时间（25维度）
        state_map[0,0,0,6:31]=self.area_time[0,0,0,:]/10
        #通讯半径内的UEs信息（28维）
        for ind,ue in enumerate(self.comm_UEs):
            if ind >=7:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,31+4*ind]=dx/100
            state_map[0,0,0,32+4*ind]=dy/100
            state_map[0,0,0,33+4*ind]=ue.F_t/5
            state_map[0,0,0,34+4*ind]=ue.trans_rate/1000
        #覆盖半径内的UEs信息（28维）
        for ind,ue in enumerate(self.Covered_UEs):
            if ind >=7:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,59+3*ind]=dx/100
            state_map[0,0,0,60+3*ind]=dy/100
            state_map[0,0,0,61+3*ind]=ue.F_t/5
            state_map[0,0,0,62+4*ind]=ue.trans_rate/1000
        state_end=time.time()
        time_p=state_end-state_start
        return copy.copy(state_map[0,0,0,:]) 
    #弱化版状态空间          
    # def state(self):
    #     #所有技能所需要的状态空间
    #     self.Min_dis=self.R
    #     state_map=np.zeros((1,1,1,87))
    #     state_map[0,0,0,0]=self.Step/150
    #     state_map[0,0,0,1]=self.position.x/500
    #     state_map[0,0,0,2]=self.position.y/500
    #     state_map[0,0,0,3]=self.V_vector.x/20
    #     state_map[0,0,0,4]=self.V_vector.y/20
    #     state_map[0,0,0,5]=self.task_executed/10000
    #     #停留时间（25维度）
    #     state_map[0,0,0,6:31]=self.area_time[0,0,0,:]/10
    #     #通讯半径内的UEs信息（28维）
    #     for ind,ue in enumerate(self.comm_UEs):
    #         if ind >=7:  #覆盖范围内可收集的ue的信息
    #             break
    #         #相对位置
    #         dx=ue.position.x-self.position.x
    #         dy=ue.position.y-self.position.y
    #         state_map[0,0,0,31+4*ind]=dx/100
    #         state_map[0,0,0,32+4*ind]=dy/100
    #         state_map[0,0,0,33+4*ind]=ue.F_t/5
    #         state_map[0,0,0,34+4*ind]=ue.trans_rate/1000
    #     #覆盖半径内的UEs信息（28维）
    #     for ind,ue in enumerate(self.Covered_UEs):
    #         if ind >=7:  #覆盖范围内可收集的ue的信息
    #             break
    #         #相对位置
    #         dx=ue.position.x-self.position.x
    #         dy=ue.position.y-self.position.y
    #         state_map[0,0,0,59+3*ind]=dx/100
    #         state_map[0,0,0,60+3*ind]=dy/100
    #         state_map[0,0,0,61+3*ind]=ue.F_t/5
    #         state_map[0,0,0,62+4*ind]=ue.trans_rate/1000
    #     return copy.copy(state_map[0,0,0,:])      

    #分层强化学习动作更新
    # ██╗  ██╗██████╗ ██████╗ ██╗     ███╗   ██╗
    # ██║  ██║██╔══██╗██╔══██╗██║     ████╗  ██║
    # ███████║██║  ██║██████╔╝██║     ██╔██╗ ██║
    # ██╔══██║██║  ██║██╔══██╗██║     ██║╚██╗██║
    # ██║  ██║██████╔╝██║  ██║███████╗██║ ╚████║
    # ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╚═══╝                                     
    def update_SHDRLN(self,action):
        #统计KL散度
        # state_test=self.state()
        # KL_cal=[]
        # SelfPolicy=self.Trainer.get_policy(state_test)
        # random_action=[]
        # # for uav in self.env.Agents:   #分别对所有其他的UAV计算策略的KL散度
        # #     OtherPolicy=uav.Trainer.get_policy(state_test)
        # #     KL_=kl_divergence(OtherPolicy,SelfPolicy)
        # #     KL_cal.append(round(KL_,2))   #保留两位小数
        # # if len(self.sum_KL)==0:
        # #     self.sum_KL=KL_cal
        # # else:
        # #     sum_ = list(map(lambda x, y: x + y, self.sum_KL, KL_cal))
        # #     self.sum_KL=sum_
        # for spn in self.SPNs:   #分别对所有其他的SPNs计算策略的KL散度
        #     state_tensor = torch.tensor([state_test], dtype=torch.float).to(device)
        #     OtherPolicy=spn(state_tensor)
        #     KL_=kl_divergence(OtherPolicy,SelfPolicy)
        #     KL_cal.append(round(KL_,2))   #保留两位小数
        #     action_dist = torch.distributions.Categorical(OtherPolicy)
        #     test_act = action_dist.sample()
        #     random_action.append(test_act)
        # if len(self.sum_KL)==0:
        #     self.sum_KL=KL_cal
        # else:
        #     sum_ = list(map(lambda x, y: x + y, self.sum_KL, KL_cal))
        #     self.sum_KL=sum_
        # # ████████╗███████╗███████╗████████╗
        # # ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
        # #     ██║   █████╗  ███████╗   ██║   
        # action=random.choice(random_action) #测试，随机选取一个SPN执行动作,不测试时请删除！！！！！！！！！！！
        # #     ██║   ██╔══╝  ╚════██║   ██║   
        # #     ██║   ███████╗███████║   ██║   
        # #     ╚═╝   ╚══════╝╚══════╝   ╚═╝      

        #action=5 #测试单个DSN的效率
        #action为选择的DSN索引
        start_SHDRLN=time.time()
        r_discount=0  #折扣奖励值之和
        r_real=0     #真实奖励之和
        self.step_skill=0   #动作步长
        #self.path.append(action)  #统计动作
        if action==0:
            #初始化部分
            while True:
                state=self.state_Skill1() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill1=self.DSNs[0](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill1)  #执行动作
                r_real=r_real+r0    
                if done:
                    break
        elif action==1:
            #初始化部分
            while True:  
                state=self.state_Skill2() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill2=self.DSNs[1](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill2)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        elif action==2:
        #else:
            #采用默认策略，只进行3步决策
            while True:  
                state=self.state_Skill3() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill3=self.DSNs[2](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill3)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        elif action==3:
            #初始化部分
            while True:  
                state=self.state_Skill4() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill4=self.DSNs[3](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill4)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        elif action==4:
            #初始化部分
            while True:  
                state=self.state_Skill5() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill5=self.DSNs[4](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill5)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        elif action==5:
            #初始化部分
            while True:  
                state=self.state_Skill6() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill5=self.DSNs[5](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill5)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        else:
            #保持神经网络与SAC一致
            while True:  
                state=self.state_Skill6() #用于技能1的状态值
                state = torch.tensor([state], dtype=torch.float).to(device)
                action_skill5=self.DSNs[5](state).data.max(1)[1].view(1, 1)   #直接通过模型得到输出
                r0,done,_=self.update_Skill(action_skill5)  #执行动作
                r_real=r_real+r0  #真实获得的奖励值之和
                if done:
                    break
        end_SHDRLN=time.time()
        time_SHDRLN=end_SHDRLN-start_SHDRLN
        #整体运行超时
        if self.Step>=self.Max_Step:
            self.done=True
            self.score+=r_real
            with open('path.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in self.path:  # 假设有10次迭代
                    csvwriter.writerow(data)
            return r_real,True,'success'
        else:
            self.score+=r_real
            return r_real,False,'normal'
    #SHDRLN整体需要的状态空间    
    def state_SHDRLN(self):
        state_start=time.time()
        #所有技能所需要的状态空间
        state_map=np.zeros((1,1,1,87))
        state_map[0,0,0,0]=self.Step/150
        state_map[0,0,0,1]=self.position.x/500
        state_map[0,0,0,2]=self.position.y/500
        state_map[0,0,0,3]=self.V_vector.x/20
        state_map[0,0,0,4]=self.V_vector.y/20
        state_map[0,0,0,5]=self.task_executed/10000
        state_map[0,0,0,6]=self.task_collect/10000
        #地图的覆盖率情况(25维度)：全局信息
        for ue in self.UEs:
            indx=int(ue.position.x/100)
            indy=int(ue.position.y/100)
            ind=indy*5+indx
            if ue.Is_covered==1:
                state_map[0,0,0,6+ind]+=0.1   #统计每个区域没有覆盖到的UEs数目
        #通讯半径内的UEs信息（15维）
        for ind,ue in enumerate(self.comm_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,32+3*ind]=dx/100
            state_map[0,0,0,33+3*ind]=dy/100
            state_map[0,0,0,34+3*ind]=ue.F_t/5
        #覆盖半径内的UEs信息（15维）
        for ind,ue in enumerate(self.Covered_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,47+3*ind]=dx/100
            state_map[0,0,0,48+3*ind]=dy/100
            state_map[0,0,0,49+3*ind]=ue.F_t/5
        #区域停留时间信息（25维）
        state_map[0,0,0,62:]=self.area_time/10
        state_end=time.time()
        time_p=state_end-state_start
        return copy.copy(state_map[0,0,0,:]) 
    #根据不同技能反馈的动作进行决策，是整体状态的更新,所有新增值变更状态都添加到这里
    def update_Skill(self,action):
        start_SHDRLN=time.time()
        #执行技能动作
        global_r=0  #全局任务奖励
        self.Step+=1    #全局步长统计
        self.step_skill+=1 #局部技能步数统计
        act=self.act[action]  #获取动作，得到加速度的大小与方向
        #速度变更
        self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
        self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
        self.V=self.Calc_V()   #计算速度大小,并自适应调整
        self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
        sum_power=self.P_fly+self.Communication_power   #计算总功率
        #判断相对位移
        self.position.x+=self.V_vector.x   #变更x坐标
        self.position.y+=self.V_vector.y    #变更y坐标
       # self.path.append(action)
        self.V_record.append(self.V)
        self.path.append([self.position.x,self.position.y,self.V])
        #更新UE任务卸载部分的状态
        if self.position.x>=self.env.len:
            self.position.x=self.env.len-1
            self.V_vector.x=(-self.V_vector.x)  #速度反转
            global_r-=1
        if self.position.x<0:
            self.position.x=0
            self.V_vector.x=(-self.V_vector.x)  #速度反转
            global_r-=1
        if self.position.y>=self.env.width:
            self.position.y=self.env.width-1
            self.V_vector.y=(-self.V_vector.y)  #速度反转
            global_r-=1
        if self.position.y<0:
            self.position.y=0
            self.V_vector.y=(-self.V_vector.y)  #速度反转
            global_r-=1
        #UE运行
        for ue in self.UEs:
            ue.run()
        #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
        self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
        self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
        self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
        self.num_comm_ue=len(self.comm_UEs)
        #更新状态
        global_r+=(-0.01)*sum_power   #能量消耗惩罚
        self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
        #统计25宫格的区域信息
        old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
        ind_x=int(self.position.x/100)   #x区域号
        ind_y=int(self.position.y/100)   #y区域号
        ind=ind_y*5+ind_x
        self.area_time[0,0,0,ind]+=1     #增加停留时间
        #无人机采集数据
        cpu_rate=0   #cpu利用率
        for ue in self.Covered_UEs:
            if ue.Now_covered==1:  #判断可收集UE
                #收集任务
                task=ue.trans()  #上传的任务量
                self.task_collect+=task     #收集的任务数目              
                #计算处理
                executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
                calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
                cpu_rate+=calcuated_task              #统计cpu利用率
                calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
                self.task_executed+=calcuated_task     #统计处理过的任务数目
                self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
                global_r+=(0.01*calcuated_task*(1+0*self.task_executed))
                #所有累积任务都处理完毕,获得额外奖励
                if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
                    ue.reset()
        #覆盖惩罚
        sum_d=len([UE for UE in self.UEs if UE.Is_covered==0])   #统计没有被覆盖到的UEs数目
        #global_r-=((sum_d/50)**2)  #未被覆盖的UE越多，惩罚越大
        #最大化cpu利用率
        #global_r-=((self.f_max-cpu_rate)/self.f_max)**2
        #UE待处理任务的惩罚
        sum_F=len([UE.F_t for UE in self.UEs])   #统计所有UEs的待处理任务
        #global_r-=((sum_F/50)**2)  #未被覆盖的UE越多，惩罚越大
        self.R_record.append(global_r)  #记录奖励大小
        end_SHDRLN=time.time()
        time_SHDRLN=end_SHDRLN-start_SHDRLN
        if self.Step>=self.Max_Step:
            return global_r,True,'success'
        elif self.step_skill>=self.step_skill_max:
            return global_r,True,'success'
        else:
            return global_r,False,'normal' 
           
       

    #技能1                                                                      
    # ███████╗██╗  ██╗██╗██╗     ██╗          ██╗
    # ██╔════╝██║ ██╔╝██║██║     ██║         ███║
    # ███████╗█████╔╝ ██║██║     ██║         ╚██║
    # ╚════██║██╔═██╗ ██║██║     ██║          ██║
    # ███████║██║  ██╗██║███████╗███████╗     ██║
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝     ╚═╝
    #执行技能1：往覆盖UEs数目最大的策略
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     #最大化cpu利用率
    #     global_r+=(self.num_covered_ue-1)   #奖励值为覆盖半径内用户数目-观测半径内的用户
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     elif self.Step%5==4:
    #         self.done=False
    #         self.score+=global_r
    #         return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 

    # #skill 1状态空间
    # def state(self):
    def state_Skill1(self):
        #所有技能所需要的状态空间
        self.Min_dis=self.R
        state_map=np.zeros((1,1,1,22))
        state_map[0,0,0,0]=self.V_vector.x/20
        state_map[0,0,0,1]=self.V_vector.y/20
        # state_map[0,0,0,2]=self.position.x/500
        # state_map[0,0,0,3]=self.position.y/500
        #通讯半径内的UEs信息（10维）
        for ind,ue in enumerate(self.comm_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,2+2*ind]=dx/100
            state_map[0,0,0,3+2*ind]=dy/100
        #覆盖半径内的UEs信息（10维）
        for ind,ue in enumerate(self.Covered_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,12+2*ind]=dx/100
            state_map[0,0,0,13+2*ind]=dy/100
        return copy.copy(state_map[0,0,0,:])   

                                                                      
    # ███████╗██╗  ██╗██╗██╗     ██╗         ██████╗     
    # ██╔════╝██║ ██╔╝██║██║     ██║         ╚════██╗    
    # ███████╗█████╔╝ ██║██║     ██║          █████╔╝    
    # ╚════██║██╔═██╗ ██║██║     ██║         ██╔═══╝     
    # ███████║██║  ██╗██║███████╗███████╗    ███████╗    
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝    ╚══════╝    
                                                                                                                                       
    #执行技能2：覆盖UEs的数据量最大
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     #最大化覆盖UEs的任务数据量
    #     sum_F=sum([UE.F_t for UE in self.Covered_UEs])
    #     global_r+=(sum_F/5-1)   
        
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
        
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     elif self.Step%5==4:
    #         self.done=False
    #         self.score+=global_r
    #         return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 
        
    # #skill 2状态空间
    def state_Skill2(self):
    #def state(self):
        #所有技能所需要的状态空间
        self.Min_dis=self.R
        state_map=np.zeros((1,1,1,34))
        state_map[0,0,0,0]=self.V_vector.x/20
        state_map[0,0,0,1]=self.V_vector.y/20
        state_map[0,0,0,2]=self.position.x/500
        state_map[0,0,0,3]=self.position.y/500
        #通讯半径内的UEs信息（15维）
        for ind,ue in enumerate(self.comm_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,4+3*ind]=dx/100
            state_map[0,0,0,5+3*ind]=dy/100
            state_map[0,0,0,6+3*ind]=ue.F_t/5
        #覆盖半径内的UEs信息（15维）
        for ind,ue in enumerate(self.Covered_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,19+3*ind]=dx/100
            state_map[0,0,0,20+3*ind]=dy/100
            state_map[0,0,0,21+3*ind]=ue.F_t/5
        return copy.copy(state_map[0,0,0,:])   
    
    # ███████╗██╗  ██╗██╗██╗     ██╗         ██████╗ 
    # ██╔════╝██║ ██╔╝██║██║     ██║         ╚════██╗
    # ███████╗█████╔╝ ██║██║     ██║          █████╔╝
    # ╚════██║██╔═██╗ ██║██║     ██║          ╚═══██╗
    # ███████║██║  ██╗██║███████╗███████╗    ██████╔╝
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝    ╚═════╝ 
    #执行技能3：控制速度使能效最大化
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     global_r+=(self.task_executed/(self.energy_cost_total+1))
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     elif self.Step%10==9:
    #         self.done=False
    #         self.score+=global_r
    #         return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 
        
    # #skill 3状态空间
    def state_Skill3(self):
    #def state(self):
        #所有技能所需要的状态空间
        self.Min_dis=self.R
        state_map=np.zeros((1,1,1,36))
        state_map[0,0,0,0]=self.V_vector.x/20
        state_map[0,0,0,1]=self.V_vector.y/20
        state_map[0,0,0,2]=self.position.x/500
        state_map[0,0,0,3]=self.position.y/500
        state_map[0,0,0,4]=(self.P_fly+self.Communication_power)/100
        state_map[0,0,0,5]=(self.task_executed)/10000
        #通讯半径内的UEs信息（15维）
        for ind,ue in enumerate(self.comm_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,6+3*ind]=dx/100
            state_map[0,0,0,7+3*ind]=dy/100
            state_map[0,0,0,8+3*ind]=ue.F_t/5
        #覆盖半径内的UEs信息（15维）
        for ind,ue in enumerate(self.Covered_UEs):
            if ind >=5:  #覆盖范围内可收集的ue的信息
                break
            #相对位置
            dx=ue.position.x-self.position.x
            dy=ue.position.y-self.position.y
            state_map[0,0,0,21+3*ind]=dx/100
            state_map[0,0,0,22+3*ind]=dy/100
            state_map[0,0,0,23+3*ind]=ue.F_t/5
        return copy.copy(state_map[0,0,0,:]) 
    
    # ███████╗██╗  ██╗██╗██╗     ██╗         ██╗  ██╗
    # ██╔════╝██║ ██╔╝██║██║     ██║         ██║  ██║
    # ███████╗█████╔╝ ██║██║     ██║         ███████║
    # ╚════██║██╔═██╗ ██║██║     ██║         ╚════██║
    # ███████║██║  ██╗██║███████╗███████╗         ██║
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝         ╚═╝
    #技能4，全局技能1，覆盖时间均衡
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         global_r-=10
    #     #UE运行
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     global_r-=(self.area_time[0,0,0,ind]/6)**2
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     elif self.Step%10==9:
    #         self.done=False
    #         self.score+=global_r
    #         return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 
        
    #skill 4状态空间
    #def state(self):
    def state_Skill4(self):
        #技能4所需状态,状态维度：13
        #将任务空间划分为9宫格，统计九个小区域的停留时长
        state_map=np.zeros((1,1,1,27))
        state_map[0,0,0,0:25]=self.area_time[0,0,0,:]
        state_map[0,0,0,25]=self.position.x/500  #收集的任务数目
        state_map[0,0,0,26]=self.position.y/500   #收集的任务数目
        return copy.copy(state_map[0,0,0,:])

    # ███████╗██╗  ██╗██╗██╗     ██╗         ███████╗
    # ██╔════╝██║ ██╔╝██║██║     ██║         ██╔════╝
    # ███████╗█████╔╝ ██║██║     ██║         ███████╗
    # ╚════██║██╔═██╗ ██║██║     ██║         ╚════██║
    # ███████║██║  ██╗██║███████╗███████╗    ███████║
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝    ╚══════╝                                             
    #技能5，全局覆盖策略2：保证每个区域收集的任务数目保持均匀,目标：所有区域的采集到的任务数目之和最大
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         #global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         #global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         #global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         #global_r-=10
    #     #UE运行
    #     sum_UEs_old=len([UE for UE in self.UEs  if UE.Is_covered==0])
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     self.area_time[0,0,0,ind]+=1     #增加停留时间
    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     #用未被覆盖的用户总数作为惩罚
    #     #sum_UEs_new=len([UE for UE in self.UEs  if UE.Is_covered==0])
    #     sum_area=self.area_time.size-np.count_nonzero(self.area_time)  #运行时间为0的个数
    #     global_r+=(25-2*sum_area)/25
    #     #global_r-=(sum_UEs_new/120)**2
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         global_r+=3*(len([UE for UE in self.UEs  if UE.Is_covered==1])/10)**2  #加上覆盖的UEs数目
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     # elif self.Step%30==9:
    #     #     self.done=False
    #     #     self.score+=global_r
    #     #     return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 
        
    # #skill 5状态空间
    #def state(self):
    def state_Skill5(self):
        #技能5所需状态
        state_map=np.zeros((1,1,1,30))
        self.area_time[self.area_time != 0] = 1   #正则化，将非0元素正则化为1
        state_map[0,0,0,0:25]=self.area_time[0,0,0,:]
        state_map[0,0,0,25]=self.position.x/500  #收集的任务数目
        state_map[0,0,0,26]=self.position.y/500   #收集的任务数目
        state_map[0,0,0,27]=self.V_vector.x/20  #收集的任务数目
        state_map[0,0,0,28]=self.V_vector.y/20   #收集的任务数目
        state_map[0,0,0,29]=len([UE for UE in self.UEs  if UE.Is_covered==1])/100  #覆盖的UEs数目 
        return copy.copy(state_map[0,0,0,:])

    # ███████╗██╗  ██╗██╗██╗     ██╗          ██████╗ 
    # ██╔════╝██║ ██╔╝██║██║     ██║         ██╔════╝ 
    # ███████╗█████╔╝ ██║██║     ██║         ███████╗ 
    # ╚════██║██╔═██╗ ██║██║     ██║         ██╔═══██╗
    # ███████║██║  ██╗██║███████╗███████╗    ╚██████╔╝
    # ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝     ╚═════╝                                            
    #技能6，全局技能，保证覆盖区域的连通性
    # def update(self,action):
    #     global_r=0  #全局任务奖励
    #     self.Step+=1    #全局步长统计
    #     act=self.act[action]  #获取动作，得到加速度的大小与方向
    #     #速度变更
    #     self.V_vector.x=self.V_vector.x+act[0]*math.cos(act[1])
    #     self.V_vector.y=self.V_vector.y+act[0]*math.sin(act[1])
    #     self.V=self.Calc_V()   #计算速度大小,并自适应调整
    #     self.P_fly=self.Calc_Fly_Power()  #计算得到飞行功率
    #     sum_power=self.P_fly+self.Communication_power   #计算总功率
    #     #判断相对位移
    #     self.position.x+=self.V_vector.x   #变更x坐标
    #     self.position.y+=self.V_vector.y    #变更y坐标
    #     self.path.append(action)
    #     self.V_record.append(self.V)
    #     #更新UE任务卸载部分的状态
    #     if self.position.x>=self.env.len:
    #         self.position.x=self.env.len-1
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         #global_r-=10
    #     if self.position.x<0:
    #         self.position.x=0
    #         self.V_vector.x=(-self.V_vector.x)  #速度反转
    #         #global_r-=10
    #     if self.position.y>=self.env.width:
    #         self.position.y=self.env.width-1
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         #global_r-=10
    #     if self.position.y<0:
    #         self.position.y=0
    #         self.V_vector.y=(-self.V_vector.y)  #速度反转
    #         #global_r-=10
    #     #UE运行
    #     sum_UEs_old=len([UE for UE in self.UEs  if UE.Is_covered==0])
    #     for ue in self.UEs:
    #         ue.run()
    #     #self.Closed_UEs = sorted(self.UEs , key=lambda ue: ue.dis2uav) #按距离大小排序
    #     self.Covered_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==1]   #筛选出可收集的
    #     self.comm_UEs  = [UE for UE in self.UEs  if UE.flag==1 and UE.Now_covered==2]   #筛选出可通讯的ue
    #     self.num_covered_ue=len(self.Covered_UEs)  #统计同时覆盖的UE数目，平均分配计算资源
    #     self.num_comm_ue=len(self.comm_UEs)
    #     #更新状态
    #     self.energy_cost_total+=sum_power  #统计消耗的能量(焦耳 J)
    #     #统计25宫格的区域信息
    #     ocuppied_area_old=np.count_nonzero(self.area_time)  #统计上一s被占领过的区域数
    #     old_area_task=copy.copy(self.area_task)  #上一时刻的区域停留时间
    #     ind_x=int(self.position.x/100)   #x区域号
    #     ind_y=int(self.position.y/100)   #y区域号
    #     ind=ind_y*5+ind_x
    #     if self.area_time[0,0,0,ind]>=0:
    #         global_r-=0.2    #待在一个区域就会受到惩罚
    #     self.area_time[0,0,0,ind]+=1     #是否停留该区域
    #     ocuppied_area_new=np.count_nonzero(self.area_time)  #统计当前被占领过的区域数
    #     global_r+=3*(ocuppied_area_new-ocuppied_area_old)   #占领区域将有奖励
    #     #设计惩罚，让占领区域周围的未被占领区域尽可能少，当前区域的邻接未占领区域尽可能多
    #     count=0   #占领区域邻接的未被占领区域
    #     for x in range(5):
    #         for y in range(5):
    #             c_ind=y*5+x
    #             if self.area_time[0,0,0,c_ind]==0:
    #                 continue
    #             if x-1>=0 and self.area_time[0,0,0,c_ind-1]==0:
    #                 count+=1
    #             if x+1<=4 and self.area_time[0,0,0,c_ind+1]==0:
    #                 count+=1
    #             if  y-1>=0 and self.area_time[0,0,0,c_ind-5]==0:
    #                 count+=1
    #             if  y+1<=4 and self.area_time[0,0,0,c_ind+5]==0:
    #                 count+=1
    #             if x-1>=0 and y-1>=0 and self.area_time[0,0,0,c_ind-6]==0:
    #                 count+=1
    #             if x+1<=4 and y-1>=0 and self.area_time[0,0,0,c_ind-4]==0:
    #                 count+=1
    #             if x-1>=0 and y+1<=4 and self.area_time[0,0,0,c_ind+4]==0:
    #                 count+=1
    #             if x+1<=4 and y+1<=4 and self.area_time[0,0,0,c_ind+6]==0:
    #                 count+=1
    #     global_r+=(ocuppied_area_new-0.25*count)/5

    #     #无人机采集数据
    #     cpu_rate=0   #cpu利用率
    #     for ue in self.Covered_UEs:
    #         if ue.Now_covered==1:  #判断可收集UE
    #             #收集任务
    #             task=ue.trans()  #上传的任务量
    #             self.task_collect+=task     #收集的任务数目              
    #             #计算处理
    #             executed_task_cycle=self.f_max/self.num_covered_ue  #分配cpu周期数
    #             calcuated_task=ue.Calc_task(executed_task_cycle)   #真实处理的任务数目（周期数）
    #             cpu_rate+=calcuated_task              #统计cpu利用率
    #             calcuated_task=calcuated_task/ue.F*ue.D       #转换成KB
    #             self.task_executed+=calcuated_task     #统计处理过的任务数目
    #             self.area_task[0,0,0,ind]+=calcuated_task    #增加区域计算过的任务数目     
    #             #global_r+=(0.03*calcuated_task*(1+0*self.task_executed))
    #             #所有累积任务都处理完毕,获得额外奖励
    #             if (ue.D_t<=ue.D and ue.F_t<=ue.F) or (ue.F_t==ue.D_t/ue.D*ue.F and ue.D_t<ue.Max_D*0.1):
    #                 ue.reset()
    #     #用未被覆盖的用户总数作为惩罚
    #     #sum_UEs_new=len([UE for UE in self.UEs  if UE.Is_covered==0])
    #     # sum_area=self.area_time.size-np.count_nonzero(self.area_time)  #运行时间为0的个数
    #     # global_r+=(25-2*sum_area)/25
    #     #global_r-=(sum_UEs_new/120)**2
    #     self.R_record.append(global_r)  #记录奖励大小
    #     if self.Step>=self.Max_Step:
    #         self.done=True
    #         #global_r+=3*(len([UE for UE in self.UEs  if UE.Is_covered==1])/10)**2  #加上覆盖的UEs数目
    #         self.score+=global_r
    #         return global_r,True,'success'
    #     elif self.Step%20==19:
    #         self.done=False
    #         self.score+=global_r
    #         return global_r,True,'normal'
    #     else:
    #         self.score+=global_r
    #         return global_r,False,'normal' 

    # #返回技能6的状态空间
    #def state(self):
    def state_Skill6(self):
        #技能6所需状态
        state_map=np.zeros((1,1,1,30))
        self.area_time[self.area_time != 0] = 1   #正则化，将非0元素正则化为1
        state_map[0,0,0,0:25]=self.area_time[0,0,0,:]
        state_map[0,0,0,26]=self.position.x/500  #收集的任务数目
        state_map[0,0,0,27]=self.position.y/500   #收集的任务数目
        state_map[0,0,0,28]=self.V_vector.x/20  #收集的任务数目
        state_map[0,0,0,29]=self.V_vector.y/20   #收集的任务数目
        return copy.copy(state_map[0,0,0,:])                        

    #航线规划
    def update_PathPlan(self,action):
         #额外的判断条件
        global_r=0  #全局任务奖励
        if len(self.sub_goals)==0:
            #到达最终目标点
            self.done=True
            global_r+=(self.Max_Step-self.Step)  
            self.score+=global_r
            self.target_step=self.Step
            return global_r,True,'success'
        #action: 速度方向变化值（-1，1），为最大转向角的比例
        self.Step+=1    #全局步长统计
        old_position=copy.copy(self.position)  #复制旧坐标
        #速度方向变更
        seta_old=calculate_angle(Loc(0,0,0),self.V_vector)  #计算速度的方位角
        dis_old=Eu_Loc_distance(self.position,self.sub_goals[0])
        dis2goal_old=Eu_Loc_distance(self.position,self.goal)
        seta_new=seta_old+action[0]*self.Steering_angle   #单位弧度
        self.V_vector.x=self.Max_V*math.cos(seta_new)
        self.V_vector.y=self.Max_V*math.sin(seta_new)
        self.V=self.Calc_V()   #计算新速度
        #坐标变更
        self.position.x+=self.V_vector.x   #变更x坐标
        self.position.y+=self.V_vector.y    #变更y坐标
        #计算速度方位和子目标终点方位的差值
        tri_goal=calculate_angle(self.position,self.sub_goals[0])
        tri_V=calculate_angle(Loc(0,0,0),self.V_vector)  #速度的方向
        #z轴坐标变化
        if action[1]<0.2:
            self.position.addZ(-0.1)
        elif action[1]>0.8:
            self.position.addZ(0.1)
        if self.env.Threaten_rate(self.position)==1:
            global_r-=0.1   #如果当前在威胁中，则给予惩罚
            self.position=old_position  #重置坐标
        dis_new=Eu_Loc_distance(self.position,self.sub_goals[0])
        dis2goal_new=Eu_Loc_distance(self.position,self.goal)
        self.path.append([self.position.x,self.position.y,self.position.z])
        self.V_record.append(Eu_Loc_distance(self.position,self.sub_goals[0]))
        #奖励值设置
        global_r-=0.13*abs(action[0])  #转向角度越大，惩罚越高
        global_r+=0.2*math.cos(abs(tri_goal-tri_V))
        global_r+=0.4*(dis_old-dis_new)  #靠近子任务目标点的奖励 
        global_r+=0.4*(dis2goal_old-dis2goal_new)   #靠近最终任务目标点的奖励 
        global_r-=0.1  #每走一步就惩罚一次
        if len(self.sub_goals)>=1:
            global_r-=0.01*abs(self.position.z-self.sub_goals[0].z)  #z轴高度差的惩罚
        self.R_record.append(global_r)  #记录奖励大小
        #统计当前航线长度,迭代次数
        self.path_len+=self.V
        self.Train_epoch+=1
        

        #动态威胁环境下的模型训练
        if self.APF_Enabled==1:
            self.Adjust_subgoal()  #调整子目标点，适用于动态环境
            total_force=self.cal_force(self.position)  #计算该目标点的受力向量
            force=Eu_Loc_distance(Loc(0,0,0),total_force)  #计算力的大小
            tri_force=calculate_angle(Loc(0,0,0),total_force)  #计算力的方位角
            global_r+=0.2*force*math.cos(abs(tri_force-tri_V))  #如果速度方向与力的方向一致，加分，否之扣分

        
        if self.Step>=self.Max_Step:
            self.done=True
            global_r+=(50-Eu_Loc_distance(self.position,self.sub_goals[0]))
            self.score+=global_r
            self.total_score+=global_r  #统计总得分
            with open('path.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in self.path:  # 假设有10次迭代
                    csvwriter.writerow(data)
            return global_r,True,'lose'
        elif Eu_Loc_distance(self.position,self.sub_goals[0])<7 or (Eu_Loc_distance(self.position,self.goal)<Eu_Loc_distance(self.sub_goals[0],self.goal)):
            #到达子目标点或者当前的位置比子目标点更加接近于目标点
            global_r+=(50-Eu_Loc_distance(self.position,self.sub_goals[0]))
            self.sub_goals.pop(0)
            if len(self.sub_goals)==0:
                #到达最终目标点
                global_r+=50
                self.done=True
                global_r+=(self.Max_Step-self.Step)  
                self.score+=global_r
                self.target_step=self.Step
                self.reach_goal=1  #统计信息，到达目标点
                self.total_score+=global_r  #统计总得分
                with open('path.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in self.path:  # 假设有10次迭代
                        csvwriter.writerow(data)
                return global_r,True,'success'
            else:
                #到达子目标目标点
                self.reset("local reset")  #局部重置
                #计算速度方位和子目标终点方位的差值
                tri_goal=calculate_angle(self.position,self.sub_goals[0])
                tri_V=calculate_angle(Loc(0,0,0),self.V_vector)
                global_r+=0.2*math.cos(abs(tri_goal-tri_V))
                global_r+=(self.Max_Step-self.Step)  
                self.score+=global_r
                self.total_score+=global_r  #统计总得分
                self.target_step=self.Step
                return global_r,True,'success'
        elif Eu_Loc_distance(self.position,self.goal)<7:
            #到达最终目标点
            self.done=True
            global_r+=50
            global_r+=(self.Max_Step-self.Step)  
            self.score+=global_r
            self.target_step=self.Step
            self.reach_goal=1  #统计信息，到达目标点
            self.total_score+=global_r  #统计总得分
            with open('path.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in self.path:  
                    csvwriter.writerow(data)
            return global_r,True,'success'
        else:
            self.score+=global_r
            self.total_score+=global_r  #统计总得分
            return global_r,False,'normal' 

    def state_PathPlan(self):
        #所有技能所需要的状态空间
        state_map=np.zeros((1,1,1,100))
        state_map[0,0,0,0]=self.Step/100
        if len(self.sub_goals)>=1:
            state_map[0,0,0,1]=(self.sub_goals[0].x-self.position.x)/10
            state_map[0,0,0,2]=(self.sub_goals[0].y-self.position.y)/10
            state_map[0,0,0,3]=(self.sub_goals[0].z-self.position.z)/10
        state_map[0,0,0,4]=self.V   #速度大小
        state_map[0,0,0,5]=self.V_vector.x
        state_map[0,0,0,6]=self.V_vector.y
        state_map[0,0,0,7]=calculate_angle(Loc(0,0,0),self.V_vector) #速度方向
        #下一个子目标点
        if len(self.sub_goals)>=2:
            state_map[0,0,0,8]=(self.sub_goals[1].x-self.position.x)/10
            state_map[0,0,0,9]=(self.sub_goals[1].y-self.position.y)/10
            state_map[0,0,0,10]=(self.sub_goals[1].z-self.position.z)/10
        #无人机附近的威胁状况，周围5*5个1m栅格的威胁情况
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+dx,self.position.y+dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,11+5*i+j]=threaten
        #无人机附近的威胁状况，周围5*5个5m栅格的威胁情况
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+5*dx,self.position.y+5*dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,36+5*i+j]=threaten
        #无人机附近的威胁状况，周围5*5个10m栅格的威胁情况
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+10*dx,self.position.y+10*dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,61+5*i+j]=threaten
        #最终终点
        state_map[0,0,0,86]=(self.goal.x-self.position.x)/10
        state_map[0,0,0,87]=(self.goal.y-self.position.y)/10
        state_map[0,0,0,88]=(self.goal.z-self.position.z)/10
        state_map[0,0,0,89]=self.position.z/10  #自身的高度
        #当前位置下5m的障碍物情况
        state_map[0,0,0,90]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-1))
        state_map[0,0,0,91]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-2))
        state_map[0,0,0,92]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-3))
        state_map[0,0,0,93]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-4))
        state_map[0,0,0,94]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-5))
        return copy.copy(state_map[0,0,0,:])