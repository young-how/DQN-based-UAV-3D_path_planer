import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练

#卷积神经网络2
class QNetwork(nn.Module):

    def __init__(self, h=100, w=100, outputs=8):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x=x.reshape(x.size(0), -1)
        # x=x.view(-1)
        x=self.head(x)
        return x

class QNetwork2(nn.Module):

    def __init__(self, h=100, w=100, outputs=8):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x=x.reshape(x.size(0), -1)
        # x=x.view(-1)
        x=self.head(x)
        return x

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

#根据参数字典param进行初始化的Q网络
class Qnet2(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, param):
        super(Qnet2, self).__init__()
        self.fc1 = torch.nn.Linear(int(param.get('w')), int(param.get('hiden_dim')))
        self.fc2 = torch.nn.Linear(int(param.get('hiden_dim')), int(param.get('output')))

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

#VA网络，用于Dueling DQN算法
class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

#VA网络，用于Dueling DQN算法(自定义环境类)
class VAnet2(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, param):
        super(VAnet2, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        if V.shape==torch.Size([1]):
            Q=V + A - A.mean(0).view(-1, 1)  # Q值由V值和A值计算得到
            Q=Q.view(-1, 1)
        else:
            Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

#VA网络，用于Dueling DQN算法(用于FANET训练环境)
class VAnet3(torch.nn.Module):
    def __init__(self, param):
        super(VAnet3, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, 2*hidden_dim)  # 共享网络部分
        self.fc2 = torch.nn.Linear(2*hidden_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        if V.shape==torch.Size([1]):
            Q=V + A - A.mean(0).view(-1, 1)  # Q值由V值和A值计算得到
            Q=Q.view(-1, 1)
        else:
            Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
    
#用于AC算法，PolicyNet与ValueNet，分别充当actor与critic
class PolicyNet(torch.nn.Module):
    def __init__(self, param):
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        self.action_dim=int(param.get('output'))
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        # x=x.view(-1,1)
        x2 = F.relu(self.fc1(x))
        x3=self.fc2(x2)
        if x3.shape==torch.Size([self.action_dim]):
            x4=F.softmax(x3, dim=0)
            if math.isnan(x4.tolist()[0]):
                print('err')
                return x4
            else: 
                return x4
        else:
            x4=F.softmax(x3, dim=1)
            if math.isnan(x4.tolist()[0][0]):
                print('err')
                return x4
            else: 
                return x4
    
#价值评估网络。输入状态，输出状态价值
class ValueNet(torch.nn.Module):
    def __init__(self, param):
        super(ValueNet, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
#DDPG算法所需网络
class PolicyNet_DDPG(torch.nn.Module):
    def __init__(self, param:dict):
        super(PolicyNet_DDPG, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        action_bound=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet_DDPG(torch.nn.Module):
    def __init__(self, param):
        super(QValueNet_DDPG, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a.view(x.size()[0], 1)], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


#SAC算法所需网络（离散动作空间）
class PolicyNet_SAC(torch.nn.Module):
    def __init__(self, param:dict):
        super(PolicyNet_SAC, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        self.action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        #x=x/100   #量级转换，防止输出的结果差距较大
        x2 = F.relu(self.fc1(x))
        x3=F.relu(self.fc2(x2))
        x5=self.fc3(x3)
        if x5.shape==torch.Size([self.action_dim]):
            x6=F.softmax(x5, dim=0)
            if math.isnan(x6.tolist()[0]):
                print('err')
                return x6
            else: 
                return x6
        else:
            x6=F.softmax(x5, dim=1)
            if math.isnan(x5.tolist()[0][0]):
                print('err')
                return x6
            else: 
                return x6
        # x = F.relu(self.fc1(x))
        # return F.softmax(self.fc2(x), dim=1)


class QValueNet_SAC(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, param:dict):
        super(QValueNet_SAC, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#SAC算法用于连续动作空间
class PolicyNetContinuous_SAC(torch.nn.Module):
    def __init__(self, param:dict):
        super(PolicyNetContinuous_SAC, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        action_bound=float(param.get('action_bound'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        std = torch.tanh(F.softplus(self.fc_std(x)))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        #log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous_SAC(torch.nn.Module):
    def __init__(self, param:dict):
        super(QValueNetContinuous_SAC, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
#PPO算法所用的神经网络
class PolicyNet_PPO(torch.nn.Module):
    def __init__(self, param:dict):
        super(PolicyNet_PPO, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class QValueNet_PPO(torch.nn.Module):
    def __init__(self, param:dict):
        super(QValueNet_PPO, self).__init__()
        state_dim=int(param.get('w'))
        hidden_dim=int(param.get('hiden_dim'))
        action_dim=int(param.get('output'))
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)