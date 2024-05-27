import heapq
import math
import sys
import os
root=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../config/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Agents') 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Envs') 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../BaseClass')
from BaseClass.CalMod import *
from Envs.Car_Driving_Env import *

class AstarPlanner():
    def __init__(self,env):
        self.env=env   #获取环境接口


    def get_neighbors(self, state):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    new_state = Loc(state.x + dx, state.y + dy, state.z + dz)
                    if self.env.check_threaten(new_state)<0.5:   #威胁小于0.5，可以插入
                        neighbors.append(new_state)
        return neighbors
    
    # 定义启发式函数（曼哈顿距离）
    def heuristic(self,state, goal):
        #return abs(state.x - goal.x) + abs(state.y - goal.y) + abs(state.z - goal.z)
        return math.sqrt(abs(state.x - goal.x)**2 + abs(state.y - goal.y)**2 + abs(state.z - goal.z)**2)

    def a_star_search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))  # (f, state)
        came_from = {}
        g_scores = {start: 0}
        f_scores = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_scores[current] + 1  # assuming a cost of 1 to move to a neighbor
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_scores[neighbor], neighbor))

        return None
if __name__ == '__main__':
    a=Loc(0,0,0)
    param={'len':100,'width':100,'h':20}
    env=Car_Driving_Env(param)
    planer=AstarPlanner(env)
    path=planer.a_star_search(Loc(2,2,2),Loc(88,88,13))  
    for p in path:
        print(f"({p.x}, {p.y}, {p.z})") 