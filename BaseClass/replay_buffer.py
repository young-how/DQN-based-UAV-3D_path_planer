######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#
import numpy as np
import random
import collections
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)  #第二种容器
        self.memory = []
        self.position = 0
       
    def push(self, batch,error):
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            del self.memory[0]    
       
    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    #第二类采样方法，将样本格式化打包
    def sample2(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done ,None,None
    
    def __len__(self):
        return len(self.memory)
    
#基于优先级的经验回放所需要的数据结构
class SumTree:
    def __init__(self, capacity: int):
        # 初始化SumTree，设定容量
        self.capacity = capacity
        # 数据指针，指示下一个要存储数据的位置
        self.data_pointer = 0
        # 数据条目数
        self.n_entries = 0
        # 构建SumTree数组，长度为(2 * capacity - 1)，用于存储树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，用于存储实际数据
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):#更新采样权重
        # 计算权重变化
        change = p - self.tree[tree_idx]
        # 更新树中对应索引的权重
        self.tree[tree_idx] = p

        # 从更新的节点开始向上更新，直到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):#向SumTree中添加新数据
        # 计算数据存储在树中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 存储数据到数据数组中
        self.data[self.data_pointer] = data
        # 更新对应索引的树节点权重
        self.update(tree_idx, p)

        # 移动数据指针，循环使用存储空间
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # 维护数据条目数
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):#采样数据
        # 从根节点开始向下搜索，直到找到叶子节点
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 如果左子节点超出范围，则当前节点为叶子节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 根据采样值确定向左还是向右子节点移动
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算叶子节点在数据数组中的索引
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])

class ReplayTree:#ReplayTree for the per(Prioritized Experience Replay) DQN. 
    def __init__(self, capacity):
        self.capacity = capacity # 记忆回放的容量
        self.tree = SumTree(capacity)  # 创建一个SumTree实例
        self.abs_err_upper = 1.  # 绝对误差上限
        self.epsilon = 0.01
        ## 用于计算重要性采样权重的超参数
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4 
        self.abs_err_upper = 1.

    def __len__(self):# 返回存储的样本数量
        return self.tree.total()
    # 经验回放模板
    # def push(self, batch):
    #     self.memory.append(batch)
    #     if len(self.memory) > self.capacity:
    #         del self.memory[0]    
       
    # def add(self, state, action, reward, next_state, done): 
    #     self.buffer.append((state, action, reward, next_state, done)) 

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    
    def push(self, sample, error):#Push the sample into the replay according to the importance sampling weight
        p = (np.abs(error.detach().numpy()) +self.epsilon) ** self.alpha
        self.tree.add(p, sample)         

    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() 

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return  batch, idxs, is_weights   #zip(*batch), idxs, is_weights

    def sample2(self, batch_size):
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() 

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        state, action, reward, next_state, done = zip(*batch)
        # idxs_re = zip(*idxs)
        # weights = zip(*is_weights)
        return np.array(state), action, reward, np.array(next_state), done,idxs,is_weights
    
    
    def batch_update(self, tree_idx, abs_errors):#Update the importance sampling weight
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
