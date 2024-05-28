import csv
from BaseClass.BaseThreaten import *
from BaseClass.CalMod import *
import random
class building(BaseThreaten):
    def __init__(self,param,env=None) :
        super().__init__(param)  
        self.position=Loc(float(param.get("position").get('x')),float(param.get("position").get('y')),float(param.get("position").get('z')))
        self.position_=self.position
        self._R=None2Value(float(param.get("_R")),10)
        self._H=None2Value(float(param.get("_H")),20)


    def reset(self):
        pass

    def reset_random(self):
        pass

    def check_threaten(self,position:Loc):
        if position.z>self._H:
            return 0
        if Eu_Loc_distance(Loc(position.x,position.y,self.position.z),self.position)<self._R:
            return 1
        else:
            return 0
    def run(self):
        pass
