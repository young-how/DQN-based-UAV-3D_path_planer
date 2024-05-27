#Double DQN算法训练器
from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
root=os.path.dirname(os.path.abspath(__file__))+'/../'  #根目录
import os
import sys
Path=os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练

class DuelingDQN_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)   #初始化基类
        self.q_local = self.NetworkFactory.Create_Network(param).to(device)  #根据参数创建Q网络
        self.q_target = self.NetworkFactory.Create_Network(param).to(device)  #根据参数创建目标网络
        self.act_num=int(param.get('output'))  #获取动作数目
        model_path=param.get('model_path')  #读取可能存在的model路径

        #创建名称
        self.name=param.get('name')
        #创建优化器
        self.optim = optim.Adam(self.q_local.parameters(), lr=self.LEARNING_RATE)   #设置优化器，使用adam优化器
        #统计变量
        self.epoch=0   #训练周期数
        self.loss=0
        self.Update_loop=int(None2Value(param.get('Update_loop'),3))

        #初始化模型
        self.Load_Mod(model_path)   #载入可能已经存在的模型参数

    #用于载入DSN模型
    def Replace_mod(self,mod,param:dict):
        self.act_num=int(param.get('output'))  #获取动作数目
        self.q_local=mod.to(device)    #替换模型
        self.q_target=mod.to(device)    #替换模型

    def Load_Mod(self,Mod_path=None):
        #如果Mod不为None，则载入模型，若为None则载入默认模型
        #如果能在 ../Mod 路径下在能找到该模型的文件，则载入模型
        if Mod_path != None:
            path_Qtarget=root+Mod_path+'q_target_DuelingDQN_'+self.name+'.pth'
            path_Qlocal=root+Mod_path+'q_local_DuelingDQN_'+self.name+'.pth'
            if os.path.exists(path_Qtarget) and os.path.exists(path_Qlocal):
                try:
                    
                    Mod_Qtarget=torch.load(path_Qtarget)
                    Mod_Qlocal=torch.load(path_Qlocal)
                    self.q_target.load_state_dict(Mod_Qtarget['model'])
                    self.q_local.load_state_dict(Mod_Qlocal['model'])
                    self.optim.load_state_dict(Mod_Qlocal['optimizer'])
                    self.epoch=Mod_Qlocal['epoch'] 
                except Exception as e:
                    print(e.args)  #输出异常信息
        else:
            path_Qtarget=root+'/Mod/q_target_DuelingDQN_'+self.name+'.pth'
            path_Qlocal=root+'/Mod/q_local_DuelingDQN_'+self.name+'.pth'
            if os.path.exists(path_Qtarget) and os.path.exists(path_Qlocal):
                try:
                    
                    Mod_Qtarget=torch.load(path_Qtarget)
                    Mod_Qlocal=torch.load(path_Qlocal)
                    self.q_target.load_state_dict(Mod_Qtarget['model'])
                    self.q_local.load_state_dict(Mod_Qlocal['model'])
                    self.optim.load_state_dict(Mod_Qlocal['optimizer'])
                    self.epoch=Mod_Qlocal['epoch'] 
                except Exception as e:
                    print(e.args)  #输出异常信息
                    

    def save(self,directory=root+'/Mod/'):
        #创建模型文件名
        filename=''
        #filename='w:'+str(self.w)+'h:'+str(self.h)+'c:'+str(self.channel)
        #存放模型到指定路径
        state = {'model': self.q_target.state_dict(), 'optimizer': self.optim.state_dict(), 'epoch': self.epoch}
        #torch.save(state, '%s/%s_local.pth' % (directory, filename))
        torch.save(state, '%s/q_target_DuelingDQN_%s.pth' % (directory,self.name))
        state = {'model': self.q_local.state_dict(), 'optimizer': self.optim.state_dict(), 'epoch': self.epoch}
        #torch.save(state, '%s/%s_target.pth' % (directory, filename))
        torch.save(state, '%s/q_local_DuelingDQN_%s.pth' % (directory,self.name))
     #根据状态选取动作
    def get_action(self,state,eps):
        #DQN框架类型
        state = torch.tensor([state], dtype=torch.float).to(device)
        sample = random.random()
        if  sample > eps or self.Is_Train==0:  #当不在训练时，直接根据神经网络选取动作值
            with torch.no_grad():
                y=self.q_local(state)
                value=y.data.max(1)[1].view(1, 1) 
                return  value   #根据Q值选择行为
        else:
            #随机选取动作
            return random.randrange(self.act_num)  #随机选取动作

    def learn_off_policy(self):
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }

        #进行训练
        if len(self.replay_memory.memory) < self.Batch_Size:
            return train_result
        #训练器参数为训练状态
        if self.Is_Train:
            transitions = self.replay_memory.sample(self.Batch_Size)  #获取批量经验数据       
            batch = Transition(*zip(*transitions))                    
            states = torch.cat(batch.state)
            #states = batch.state
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            next_states = torch.cat(batch.next_state)
            dones = torch.cat(batch.done)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to newtork q_local (current estimate)
            Q_expected = self.q_local(states).gather(1, actions)     #获得Q估计值
            max_action = self.q_local(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.q_target(next_states).gather(1, max_action)
        
            #Q_targets_next = self.q_target(next_states).gather(1, actions) 
            #Q_targets_next = self.q_target(next_states).gather(1, actions)    #计算Q目标值估计 #注：修改版写法

            # Compute the expected Q values
            Q_targets = rewards + (self.gamma * max_next_q_values * (1-dones))   #更新Q目标值
            #训练Q网络
            
            #loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))  #计算误差
            loss = self.mse_loss(Q_expected, Q_targets)  #计算误差
            self.optim.zero_grad()
            # backpropagation of loss to NN        
            loss.backward()
            self.loss = loss
            self.optim.step()

        #训练次数到达更新周期，进行更新目标网络
        if self.epoch%self.Update_loop==0:
            self.hard_update()
        
        #训练次数到达保存周期，保存目标网络和Q网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        return train_result  #返回训练结果
    
    #通用型更新方式
    def update(self,transition_dict):
        #离线训练
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }
        #没有训练数据，返回
        if transition_dict['states']==[]:
            return train_result  #返回训练结果
        #训练器参数为训练状态
        if self.Is_Train:
            states = torch.tensor(transition_dict['states'],dtype=torch.float).to(device)
            actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1, 1).to(device)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
            dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)
            Q_expected = self.q_local(states).gather(1, actions.long())     #获得Q估计值
            #rewards=(rewards+3)/3
            max_action = self.q_local(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.q_target(next_states).gather(1, max_action)
            #Q_targets_next = self.q_target(next_states).gather(1, actions) 
            #Q_targets_next = self.q_target(next_states).gather(1, actions)    #计算Q目标值估计 #注：修改版写法
            # Compute the expected Q values
            Q_targets = rewards + (self.gamma * max_next_q_values * (1-dones))   #更新Q目标值
            #训练Q网络
            #loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))  #计算误差
            loss = self.mse_loss(Q_expected, Q_targets)  #计算误差
        
            self.optim.zero_grad()
            # backpropagation of loss to NN        
            loss.backward()
            self.loss = loss
            self.optim.step()

        #训练次数到达更新周期，进行更新目标网络
        if self.epoch%self.Update_loop==0:
            self.hard_update()
        
        #训练次数到达保存周期，保存目标网络和Q网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        return train_result  #返回训练结果
    #返回策略分布
    def get_policy(self,state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.q_local(state)
        max_index = torch.argmax(probs)
        new_tensor = torch.zeros_like(probs)
        new_tensor[0][max_index] = 1
        return  new_tensor
    def hard_update(self):
        #更新目标网络
        for target_param, param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(param.data)
    
    def replace_param(self,target):
        #根据目标网络替换参数
        for target_param, param in zip(target.parameters(), self.q_local.parameters()):
            param.data.copy_(target_param.data)
    
    def replace_target_param(self,target):
        #根据目标网络替换参数
        for target_param, param in zip(target.parameters(), self.q_target.parameters()):
            param.data.copy_(target_param.data)
 
    def Push_Replay(self,Experience):
        #将经验存放到经验池中
        self.replay_memory.push(Experience)

    def set_replay_size(self,replay_size:int):
        #设置经验池大小
        self.replay_size=replay_size

    def set_LEARNING_RATE(self, LEARNING_RATE: float):
        # 设置学习率
        self.LEARNING_RATE = LEARNING_RATE

    def set_Batch_Size(self, Batch_Size: int):
        # 设置批量大小
        self.Batch_Size = Batch_Size

    def set_gamma(self, gamma: float):
        # 设置折扣率
        self.gamma = gamma

    def set_max_epoch(self, max_epoch: int):
        # 设置最大训练周期数
        self.max_epoch = max_epoch

    def set_save_loop(self, save_loop: int):
        # 设置模型参数保存周期
        self.save_loop = save_loop