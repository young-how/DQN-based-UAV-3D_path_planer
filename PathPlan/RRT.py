import os
import time
import numpy as np
import random
import math
import sys
root=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../config/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Agents') 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Envs') 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from BaseClass.CalMod import *
class Node:
    def __init__(self, loc):
        self.loc = loc
        self.parent = None
        self.cost = 0.0
class RRTPlanner:
    def __init__(self, env):
        self.env = env
        self.step_size=10  #默认的步长
        self.iter_num=10000 #迭代周期
        self.obstacle_step=5 

    def get_random_point(self,goal):
        if random.uniform(0,1)>0.5:
            return Loc(
                random.uniform(0, self.env.len),
                random.uniform(0, self.env.width),
                random.uniform(0, self.env.h)
            )
        else:
            return goal

    def nearest_node(self, nodes, point):
        return min(nodes, key=lambda node: node.loc.distance(point))

    def steer(self, from_node, to_point, step_size):
        direction = np.array([to_point.x - from_node.loc.x, to_point.y - from_node.loc.y, to_point.z - from_node.loc.z])
        length = np.linalg.norm(direction)
        if length < step_size:
            return Loc(to_point.x, to_point.y, to_point.z)
        direction = direction / length
        new_point = np.array([from_node.loc.x, from_node.loc.y, from_node.loc.z]) + direction * step_size
        return Loc(new_point[0], new_point[1], new_point[2])

    def obstacle_free(self, start, end, step_size=1.0):
        steps = int(start.distance(end) / step_size)
        for i in range(steps + 1):
            x = start.x + (end.x - start.x) * i / (steps+1)
            y = start.y + (end.y - start.y) * i / (steps+1)
            z = start.z + (end.z - start.z) * i / (steps+1)
            if self.env.Threaten_rate(Loc(x, y, z)) == 1:
                return False
        return True
    def Set_StepSize(self,L):
        self.step_size=L
    def Set_IterNum(self,n):
        self.iter_num=n
    def Set_OBStep(self,n):
        self.obstacle_step=n
    def getPath(self, start:Loc, goal:Loc):
        max_iter = self.iter_num
        step_size = self.step_size
        goal_radius = self.step_size
        obstacle_detect_R=self.obstacle_step

        start_node = Node(start)
        goal_node = Node(goal)
        nodes = [start_node]

        for _ in range(max_iter):
            rand_point = self.get_random_point(goal)
            nearest = self.nearest_node(nodes, rand_point)
            new_loc = self.steer(nearest, rand_point, step_size)
            new_node = Node(new_loc)

            if not self.obstacle_free(nearest.loc, new_loc,obstacle_detect_R):
                continue

            new_node.parent = nearest
            new_node.cost = nearest.cost + nearest.loc.distance(new_loc)
            nodes.append(new_node)

            for node in nodes:
                if node == new_node:
                    continue
                if node.loc.distance(new_loc) < step_size and new_node.cost > node.cost + node.loc.distance(new_loc):
                    if self.obstacle_free(node.loc, new_loc,obstacle_detect_R):
                        new_node.parent = node
                        new_node.cost = node.cost + node.loc.distance(new_loc)

            if new_node.loc.distance(goal) <= goal_radius:
                goal_node.parent = new_node
                break

        path = []
        node = goal_node
        while node:
            path.append(node.loc)
            node = node.parent
        path.reverse()

        return path,path
# 假设的环境类
class Environment:
    def __init__(self, len, width, h):
        self.len = len
        self.width = width
        self.h = h

    def Threaten_rate(self, loc):
        # 示例函数，返回1表示是障碍物，返回0表示不是障碍物
        return 0  # 示例中，所有点都不是障碍物
# 测试
if __name__ == "__main__":
    env = Environment(500, 500, 100)
    planner = RRTPlanner(env)
    
    start = Loc(5, 5, 5)
    goal = Loc(400, 400, 50)
    starttime=time.time()
    path = planner.getPath(start, goal)
    endtime=time.time()
    print(endtime-starttime)
    for loc in path:
        print(f"({loc.x}, {loc.y}, {loc.z})")