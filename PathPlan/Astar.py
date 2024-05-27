import heapq
import math

# 定义三维状态类
class State:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other):
        return False  # 始终返回False，表示不可比较

# 定义三维环境类
class Environment:
    def __init__(self, grid):
        self.grid = grid
        self.num_rows = len(grid)
        self.num_cols = len(grid[0])
        self.num_levels = len(grid[0][0])

    def is_valid_position(self, state):
        x, y, z = state.x, state.y, state.z
        return 0 <= x < self.num_rows and 0 <= y < self.num_cols and 0 <= z < self.num_levels and self.grid[x][y][z] == 0

    def get_neighbors(self, state):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    new_state = State(state.x + dx, state.y + dy, state.z + dz)
                    if self.is_valid_position(new_state):
                        neighbors.append(new_state)
        return neighbors

# 定义启发式函数（曼哈顿距离）
def heuristic(state, goal):
    #return abs(state.x - goal.x) + abs(state.y - goal.y) + abs(state.z - goal.z)
    return math.sqrt(abs(state.x - goal.x)**2 + abs(state.y - goal.y)**2 + abs(state.z - goal.z)**2)

# 实现A*算法
def a_star_search(env, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # (f, state)
    came_from = {}
    g_scores = {start: 0}
    f_scores = {start: heuristic(start, goal)}

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

        for neighbor in env.get_neighbors(current):
            tentative_g_score = g_scores[current] + 1  # assuming a cost of 1 to move to a neighbor
            if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_scores[neighbor], neighbor))

    return None

# 示例使用
grid = [
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
]

start_state = State(0, 0, 0)
goal_state = State(2, 2, 2)

env = Environment(grid)
path = a_star_search(env, start_state, goal_state)

if path is None:
    print("No path found")
else:
    print("Path found:")
    for state in path:
        print(f"({state.x}, {state.y}, {state.z})")
