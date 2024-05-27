from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
import sys
import os
root=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../config/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Threatens') 
from Threatens.RD import *
import random
class Car(BaseAgent):
    def __init__(self,param:dict,env=None) -> None:
        #初始化
        #输入参数：
        #       param   :   初始化参数字典
        super().__init__(param)   #初始化基类
        #自定义附加参数添加在如下地方：
        self.param=param  #保存初始化参数信息
        #car特有属性
        self.V_vector=Loc(0,0,0)  #速度矢量
        self.Max_V=int(param.get("Max_V"))   #最大速度
        self.R=int(param.get("R"))   #探测范围
        self.Acceration=int(param.get("Acceration"))   #加速度大小
        self.Max_Step=int(param.get("Max_Step"))   #加速度大小
        self.Step=0
        #car动作空间
        #self.act=[0,0.25*math.pi,0.5*math.pi,0.75*math.pi,math.pi,1.25*math.pi,1.5*math.pi,1.75*math.pi]
        self.act_num=int(param.get("act_num"))   #动作空间大小
        self.act=[]
        for i in range(self.act_num):
            self.act.append(2*math.pi*i/self.act_num)

        self.env=env  #设置所处的环境

        #构建状态图层
        self.map_w=int(param.get("map_w"))  #获取状态图层宽度
        self.map_h=int(param.get("map_h"))  #获取状态图长度
        self.map_c=int(param.get("map_c"))  #获取状态图层通道数
        self.Min_dis=self.R  #最小距离障碍物的距离

        #目标点
        self.goal=Loc(75,75,0)
        self.reach_r=Eu_Loc_distance(self.position,self.goal)  #到达终点的奖励
        #状态总图层
        self.state_map=np.zeros((1,self.map_c,self.map_h,self.map_w))
        self.state_map.fill(self.R)

        #障碍物探测向量
        self.ob_map=np.ones((1,self.act_num))


    def Set_Env(self,env):
        #设置所属环境类
        self.env=env


    def reset(self):
        self.Step=0
        self.score=0
        self.done=False
        self.position=Loc(int(self.param.get("position").get('x')),int(self.param.get("position").get('y')),int(self.param.get("position").get('z')))  #初始化在空间中的位置
        self.V_vector=Loc(0,0,0)  #速度矢量

        #目标点
        # self.goal=Loc(random.randint(1,99),random.randint(1,99),0)
        # seta=random.uniform(0,2*math.pi)

        #存在两种目标点
        self.goal=Loc(80,80,0)  #测试用

        #添加障碍物
        # self.env.Threatens=[]
        # self.env.Threatens.append(RD(Loc(65,65,0),10))

        # if random.uniform(0,1)<0.5:   #训练用
        #     self.goal=Loc(25,75,0)
        # else:
        #     self.goal=Loc(75,25,0)



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
    
    def update(self,action):
        #该函数给定动作编号，使智能体完成该动作，从而更新并返回智能体状态
        old_p=self.position  #保存旧路径点
        r=0
        self.Step+=1
        act=self.act[action]  #动作值，速度变更情况
        delta_V=Loc(self.Acceration*math.cos(act),self.Acceration*math.sin(act),0)  #速度变更矢量
        self.V_vector=Loc(self.V_vector.x+delta_V.x,self.V_vector.y+delta_V.y,0)  #变更后的速度
        V=math.sqrt(self.V_vector.x**2+self.V_vector.y**2)  #当前速度
        if V>self.Max_V:
            #超过最大速度，约束速度
            scale=self.Max_V/V
            self.V_vector=Loc(self.V_vector.x*scale,self.V_vector.y*scale,0)

        #位移前相对终点的距离
        ds=Eu_Loc_distance(self.position,self.goal)

        #判断相对位移
        self.position.x+=self.V_vector.x   #变更x坐标
        self.position.y+=self.V_vector.y   #变更y坐标

        if self.env.check_threaten(self.position)<0.5:  #路径点可走
            #位移后相对终点的距离
            ds2=Eu_Loc_distance(self.position,self.goal)

            r+=(ds-ds2)
        else:   #路径点不可走
            self.position=old_p   #恢复路径点
            r-=0.1   #误判惩罚

        #更新障碍物向量
        detect_R=10  #探测距离
        for dir,indx in enumerate(self.act):
            R=detect_R
            while R>=0:
                px=self.position.x+R*math.cos(dir)
                py=self.position.y+R*math.sin(dir)
                pz=self.position.z
                if self.env.check_threaten(Loc(px,py,pz))<0.5:
                    break
                R-=1
            self.ob_map[0,int(indx)]=R/detect_R   #归一化
            



        #更新状态
        #self.state()
        r+=(-0.01)
        if Eu_Loc_distance(self.position,self.goal)<=3:
            self.done=True
            self.score+=(r+10-Eu_Loc_distance(self.position,self.goal))
            return (r+10-Eu_Loc_distance(self.position,self.goal)),True,'success'
        elif self.Step>=self.Max_Step:
            #完成任务
            self.done=True
            self.score+=(r+10-Eu_Loc_distance(self.position,self.goal))
            return (r+10-Eu_Loc_distance(self.position,self.goal)),True,'lose'
        else:
            self.score+=r
            return r,False,'normal'
        

        
         
    
    def state(self):
        self.Min_dis=self.R
        self.state_map.fill(0)

        self.state_map[0,0,0,0]=abs(Eu_Loc_distance(self.V_vector,Loc(0,0,0)))  #速度大小
        self.state_map[0,0,0,1]=calculate_angle(Loc(0,0,0),self.V_vector)   #速度方向
        self.state_map[0,0,0,2]=Eu_Loc_distance(self.position,self.goal)  #距离终点距离
        self.state_map[0,0,0,3]=calculate_angle(self.position,self.goal)  #距离终点方向
        self.state_map[0,0,0,4]=self.Step
        
        #更新障碍物向量
        # self.state_map[0,0,0,5:]=self.ob_map[0,:]
        return self.state_map[0,0,0,:]
    def Go_AstarPath(self):
        #通过astar路径更新状态


        r=0   #奖励值


        
        #判断相对位移
        x_old=self.position.x
        y_old=self.position.y
        self.position.x=self.Astar_path[0].x
        self.position.y=self.Astar_path[0].y
        self.Astar_path.pop(0)  #删除第一个元素

        dx=self.position.x-x_old
        dy=self.position.y-y_old
        action=self.act.index([dx,dy])  #查找该动作所在的动作编号



        #判断相对位移
        dx_now=abs(self.goal.x-self.position.x)
        dy_now=abs(self.goal.y-self.position.y)
        Mah_dis_now=dx_now+dy_now

        #更新方位信息
        self.Threaten_rate=self.env.check_total_threaten(self.position.x,self.position.y,0)       #当前威胁概率
        self.path.append(path_point(self.position.x,self.position.y,self.position.z,0))    #航线点
        self.distance_his+=1
        dis_old=self.distance_goal  #上一段时间的距离终点的距离
        self.distance_goal=Eu_distance(self.position.x,self.position.y,self.position.z,self.goal.x,self.goal.y,self.goal.z)   #当前与终点的欧式距离

        #更新路线图层，并判别是否走过历史路径
        if self.Route_map[0,0,self.position.x,self.position.y]==-1:
            r-=1
        self.Update_Route_map()
        
        #障碍物图层奖励
        if self.Threaten_rate==1:
            r=r-1

        #APF图层奖励
        r+=(self.APF_map[0,0,self.position.x,self.position.y]-self.APF_map[0,0,x_old,y_old])


        reach_dis=2
        if self.distance_goal<=reach_dis or self.Astar_path==[]:
            #到达目标点,避免稀疏奖励，难度等级低时增大判定范围
            #discount=(self.Dis_Start2Goal/self.distance_his)  #到达目标点的奖励折扣
            discount2=1
            #reach_reward=200
            #r=r+10+10*(1-self.step/self.step_max)**2  #固定奖励+附加奖励，附加奖励根据所走步数确定
            r=300   #固定奖励
            self.stat=1  #状态
            self.score+=r
            self.done=True
            return action,r,True,1

        if self.step>=self.step_max:
            self.done=True
            #discount=(self.distance_goal/self.Dis_Start2Goal)  #超时的奖励折扣
            #r=r+10/(self.distance_goal*self.distance_goal)
            r=0  #固定奖励
            self.score+=r  
            self.stat=2  #状态
            return action,r,True,5   #超过任务时长
        if self.Threaten_rate==1:
            r=-1  #固定奖励
            return action,r,False,2   #当前在威胁区飞行
        self.stat=0
        return action,r,False,4  #[奖励，是否终止状态，状态编号]

