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
        self.Planner.Set_StepSize(self.sub_granularity)  #设置RRT的格子长度
        Path,path_ori=self.Planner.getPath(self.position,goal)  #得到栅格路径，还需要将之转换为真实路径
        if Path==None:
            #没有得到路径
            return None,None
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
            x=random.uniform(10,210)
            y=random.uniform(1,10)
            self.position=Loc(x,y,0)
            x=random.uniform(330,490)
            y=random.uniform(420,490)
            self.goal=Loc(x,y,0)
            #self.goal=Loc(480,480,25)  #固定目标点
            _,Astar_path=self.Cal_SubTask(self.goal)  #子任务拆分
            #训练统计数据，存放到数据库中
            self.total_score=0   #整个任务的得分
            self.path_len=0     #当前航线的长度
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
        if self.env.Threaten_rate(self.position)==1:
            global_r-=0.3   #如果当前在威胁中，则给予惩罚
            self.position=old_position  #重置坐标
            tri_V=calculate_angle(self.position,self.sub_goals[0])
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