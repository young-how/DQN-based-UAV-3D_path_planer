#基本运算的模型库与需要的类
from typing import Union
import sys
#sys.path.append("E:\younghow\RLGF")
import torch
import xmltodict
import math

#基础的设备配置
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
device = torch.device("cpu")    #使用cPU进行训练
FloatTensor =  torch.FloatTensor  #cpu
threaten_threshold=0.7  #威胁阈值
class Loc():
    def __init__(self,x,y,z) -> None:
        self.x=x
        self.y=y
        self.z=z
    def Set_Value(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    def Copy_From(self,p2):
        self.x=p2.x
        self.y=p2.y
        self.z=p2.z
    def __eq__(self, other):
        if other==None:
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __add__(self, other):
        if not isinstance(other, Loc):
            raise ValueError("Can only add a Loc object with another Loc object")
        new_x = self.x + other.x
        new_y = self.y + other.y
        new_z = self.z + other.z
        return Loc(new_x, new_y, new_z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other):
        return False  # 始终返回False，表示不可比较

def None2Value(value1,value2=None):
    #如果value1不为None，返回value1，否则返回value2
    if value1 ==None:
        return value2
    else:
        return value1

#计算Loc之间欧式距离
def Eu_Loc_distance(loc1,loc2):
    return math.sqrt((loc1.x-loc2.x)**2+(loc1.y-loc2.y)**2+(loc1.z-loc2.z)**2)

def Eu_distance(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

#路径点信息
class path_point:
    def __init__(self,x,y,z,t) :
       self.point=Loc(x,y,z)
       self.Time_Stamp=t

#计算两点连线与x轴的夹角
def calculate_angle(x1, y1, x2, y2):
    # calculate difference in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1
    # calculate angle using arctan2 function
    angle = math.atan2(dy, dx)
    # convert angle from radians to degrees
    angle = math.degrees(angle)
    # normalize to range [0, 360)
    return (angle + 360) % 360

#计算两点连线与x轴的夹角 p1->p2
def calculate_angle(p1:Loc,p2:Loc,mod=1):
    #mod:  0:输出角度   1：输出弧度
    # calculate difference in x and y coordinates
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    # calculate angle using arctan2 function
    angle = math.atan2(dy, dx)
    # convert angle from radians to degrees
    angle = math.degrees(angle)
    # normalize to range [0, 360)
    if mod==0:    
        return (angle + 360) % 360
    else:
        return (angle + 360) % 360/180*math.pi
#计算两个向量的夹角是钝角还是锐角
def calculate_vec_angle(p1:Loc,p2:Loc):
    start=Loc(0,0,0)
    seta1=calculate_angle(start,p1)
    seta2=calculate_angle(start,p2)
    delta=math.cos(abs(seta1-seta2))
    if delta>0:
        return 1   #锐角
    elif delta<0:
        return -1 #钝角
    else:
        return 0 #方向一致


def XML2Dict(file_path):
    #输入xml文件的路径，将其解析并返回dict
    with open(file_path, 'r') as f:
        xml_data = f.read()

    # 将XML转换成字典
    dict_data = xmltodict.parse(xml_data)
    return dict_data

#功率转换公式
def dbm2watt(dbm):
    return 10**(dbm / 10)
def watt2dbm(watt):
    return 10 * math.log10(watt / 0.001)

#计算多个航线点组成的航线的长度
def calculate_path_len(path):
    len=0
    pre_p=path.pop(0)
    for p in path:
        len+=Eu_Loc_distance(pre_p,p)
        pre_p=p
    return len

def float_range(start: int, end: int, step: Union[int, float], round_number: int = 2):
    """
    浮点数 range
    :param start: 起始值
    :param end: 结束值
    :param step: 步长
    :param round_number: 精度
    :return: 返回一个 list
    """
    temp = []
    while True:
        if start < end:
            temp.append(round(start, round_number))
            start += step
        else:
            break
    return temp

def surface3d_data(th):
    for t0 in float_range(th.position.x-th._R-3, th.position.x+th._R+3, 3):
        x = t0
        for t1 in float_range(th.position.y-th._R-3, th.position.y+th._R+3, 3):
            if (t0-th.position.x)**2+(t1-th.position.y)**2>th._R**2:
                z=0
            else:
                z=math.sqrt(th._R**2 - ((t0-th.position.x)**2+(t1-th.position.y)**2)) if ((t0-th.position.x)**2+(t1-th.position.y)**2) <= th._R**2 else 0
            y = t1
            #z = th._R**2-((t0-th.position.x)**2+(t1-th.position.y)**2)
            yield [x, y, z]

if __name__=="__main__":
    p1=Loc(1,0,0)
    p2=Loc(1,1,0)
    print(calculate_vec_angle(p1,p2))