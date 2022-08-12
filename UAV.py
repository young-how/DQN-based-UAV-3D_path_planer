######################################################################
# UAV Class
#---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
#UAV 类描述，对无人机的状态参数进行初始化，
# 包括坐标、目标队列、环境、电量、方向、基础能耗、当前能耗、已经消耗能量、
# 侦测半径、周围障碍情况、坠毁概率、距离目标距离、已走步长等。
# 成员函数能返回自身状态，并根据决策行为对自身状态进行更新。
#----------------------------------------------------------------
# UAV class description, initialize the state parameters of the UAV, 
# including coordinates, target queue, environment, power, direction, 
# basic energy consumption, current energy consumption, consumed energy, 
# detection radius, surrounding obstacles, crash probability, distance Target distance, steps taken, etc. 
# Member functions can return their own state and update their own state according to the decision-making behavior.
#################################################################
import math
import random

class UAV():
    def __init__(self,x,y,z,ev):
        #初始化无人机坐标位置
        self.x=x
        self.y=y
        self.z=z
        #初始化无人机目标坐标
        self.target=[ev.target[0].x,ev.target[0].y,ev.target[0].z]
        self.ev=ev  #无人机所处环境
        #初始化无人机运动情况
        self.bt=5000  #无人机电量
        self.dir=0   #无人机水平运动方向，八种情况(弧度)
        self.p_bt=10   #无人机基础能耗，能耗/步
        self.now_bt=4   #无人机当前状态能耗
        self.cost=0   #无人机已经消耗能量
        self.detect_r=5  # 无人机探测范围 （格）
        self.ob_space=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  #无人机邻近栅格障碍物情况
        self.nearest_distance=10  #最近障碍物距离
        self.dir_ob=None     #距离无人机最近障碍物相对于无人机的方位
        self.p_crash=0   #无人机坠毁概率
        self.done=False   #终止状态
        self.distance=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])   #无人机当前距离目标点曼哈顿距离
        self.d_origin=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])   #无人机初始状态距离终点的曼哈顿距离
        self.step=0           #无人机已走步数
    def cal(self,num):
        #利用动作值计算运动改变量
        if num==0:
            return -1
        elif num==1:
            return 0
        elif num==2:
            return 1
        else:
            raise NotImplementedError
    def state(self):
        dx=self.target[0]-self.x
        dy=self.target[1]-self.y
        dz=self.target[2]-self.z
        state_grid=[self.x,self.y,self.z,dx,dy,dz,self.target[0],self.target[1],self.target[2],self.d_origin,self.step,self.distance,self.dir,self.p_crash,self.now_bt,self.cost]
        #更新临近栅格状态
        self.ob_space=[]
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i==0 and j==0 and k==0:
                        continue
                    if self.x+i<0 or self.x+i>=self.ev.len or self.y+j<0 or self.y+j>=self.ev.width or self.z+k<0 or self.z+k>=self.ev.h:
                        self.ob_space.append(1) 
                        state_grid.append(1)
                    else:
                        self.ob_space.append(self.ev.map[self.x+i,self.y+j,self.z+k])  #添加无人机临近各个栅格状态
                        state_grid.append(self.ev.map[self.x+i,self.y+j,self.z+k])
        return state_grid
    def update(self,action):
        #更新无人机状态
        dx,dy,dz=[0,0,0]
        temp=action
        #相关参数
        b=3   #撞毁参数 原3
        wt=0.005   #目标参数
        wc=0.07  #爬升参数 原0.07
        we=0   #能量损耗参数  原0.2
        c=0.05   #风阻能耗参数 原0.05
        crash=0   #坠毁概率惩罚增益倍数 原3
        Ddistance=0   #距离终点的距离变化量

        
        #计算无人机坐标变更值
        dx=self.cal(temp%3)
        temp=int(temp/3)
        dy=self.cal(temp%3)
        temp=int(temp/3)
        dz=self.cal(temp)
        #如果无人机静止不动，给予大量惩罚
        if dx==0 and dy==0 and dz==0:
            return -1000,False,False
        self.x=self.x+dx
        self.y=self.y+dy
        self.z=self.z+dz
        Ddistance=self.distance-(abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2]))  #正代表接近目标，负代表远离目标
        self.distance=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2]) #更新距离值
        self.step+=abs(dx)+abs(dy)+abs(dz)

        flag=1
        if abs(dy)==dy:
            flag=1
        else:
            flag=-1

        if dx*dx+dy*dy!=0:
            self.dir=math.acos(dx/math.sqrt(dx*dx+dy*dy))*flag     #无人机速度方向（弧度）

        #计算能耗与相关奖励
        self.cost=self.cost+self.now_bt   #更新能量损耗状态
        a=abs(self.dir-self.ev.WindField[1])    #无人机速度方向与风速方向夹角
        self.now_bt=self.p_bt+c*self.ev.WindField[0]*(math.sin(a)-math.cos(a))   #计算当前能耗
        #r_e=-we*math.exp((self.cost+self.now_bt)/self.bt)
        r_e=we*(self.p_bt-self.now_bt)   
        
        #计算碰撞概率与相应奖励
        r_ob=0
        for i in range(-2,3):
            for j in range(-2,3):
                if i==0 and j==0:
                    continue  #排除无人机所在点
                if self.x+i<0 or self.x+i>=self.ev.len or self.y+j<0 or self.y+j>=self.ev.width or self.z<0 or self.z>=self.ev.h:
                        continue  #超出边界，忽略
                if self.ev.map[self.x+i,self.y+j,self.z]==1 and abs(i)+abs(j)<self.nearest_distance:
                    self.nearest_distance=abs(i)+abs(j)
                    flag=1
                    if abs(j)==-j:
                        flag=-1
                    self.dir_ob=math.acos(i/(i*i+j*j))*flag  #障碍物相对于无人机方向
        #计算坠毁概率
        if self.nearest_distance>=4 or self.ev.WindField[0]<=self.ev.v0:
            self.p_crash=0
        else:
            #根据公式计算撞毁概率
            self.p_crash=math.exp(-b*self.nearest_distance*self.ev.v0*self.ev.v0/(0.5*math.pow(self.ev.WindField[0]*math.cos(abs(self.ev.WindField[1]-self.dir_ob)-self.ev.v0),2)))
            #self.p_crash=0
        
        
        
        #计算爬升奖励
        r_climb=-wc*(abs(self.z-self.target[2]))
        #计算目标奖励
        #r_target=-wt*(abs(self.x-self.target[0])+abs(self.y-self.target[1]))   #奖励函数1
        #r_target=Ddistance                                                     #奖励函数2
        if self.distance>1:
            r_target=2*(self.d_origin/self.distance)*Ddistance                #奖励函数3越接近目标，奖励越大
        else:
            r_target=2*(self.d_origin)*Ddistance 
        #计算总奖励
        r=r_climb+r_target+r_e-crash*self.p_crash   
        #终止状态判断
        if self.x<=0 or self.x>=self.ev.len-1 or self.y<=0 or self.y>=self.ev.width-1 or self.z<=0 or self.z>=self.ev.h-1 or self.ev.map[self.x,self.y,self.z]==1 or random.random()<self.p_crash:
            #发生碰撞，产生巨大惩罚
            return r-200,True,2
        if self.distance<=5:
            #到达目标点，给予f大量奖励
            #self.ev.map[self.x,self.y,self.z]=0
            return r+200,True,1
        if self.step>=self.d_origin+2*self.ev.h:
            #步数超过最差步长，给予惩罚
            return r-20,True,5
        if self.cost>self.bt:
            #电量耗尽，给予大量惩罚
            return r-20,True,3
        return r,False,4