#DDPG算法训练器
from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
root=os.path.dirname(os.path.abspath(__file__))+'/../'  #根目录
import os
import sys
Path=os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)


class DDPG_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)   #初始化基类
        #创建名称
        self.name=param.get('name')
        actor_param=param.get('actor')
        critic_param=param.get('critic')
        actor_lr=float(actor_param.get('lr'))     #actor学习率
        critic_lr=float(critic_param.get('lr'))  #critic学习率
        
        self.actor = self.NetworkFactory.Create_Network(actor_param).to(device)  #根据参数创建Q网络
        self.critic  = self.NetworkFactory.Create_Network(critic_param).to(device)  #根据参数创建目标网络
        #目标网络
        self.target_actor = self.NetworkFactory.Create_Network(actor_param).to(device)  #根据参数创建Q网络
        self.target_critic  = self.NetworkFactory.Create_Network(critic_param).to(device)  #根据参数创建目标网络

        #创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)   #设置优化器，使用adam优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)   #设置优化器，使用adam优化器
        #DDPG训练超参数

        #self.sigma = float(None2Value(param.get('sigma'),0))  # 高斯噪声的标准差,均值直接设为0
        self.tau =float(None2Value(param.get('tau'),0))   # 目标网络软更新参数

        #统计变量
        self.epoch=0   #训练周期数
        self.loss=0
        self.Update_loop=int(None2Value(param.get('Update_loop'),3))

        #初始化模型
        self.Load_Mod()   #载入可能已经存在的模型参数

    def Load_Mod(self,Mod=None):
        #如果Mod不为None，则载入模型，若为None则载入默认模型
        #如果能在 ../Mod 路径下在能找到该模型的文件，则载入模型
        if Mod != None:
            self.actor.load_state_dict(Mod['actor_model'])
            self.critic.load_state_dict(Mod['critic_model'])
            self.optim.load_state_dict(Mod['optimizer'])
            self.epoch=Mod['epoch'] 
        else:
            if self.name==None:
                path_actor=root+'/Mod/DDPG_actor.pth'
                path_critic=root+'/Mod/DDPG_critic.pth'
            else:
                path_actor=root+'/Mod/DDPG_actor_'+self.name+'.pth'
                path_critic=root+'/Mod/DDPG_critic_'+self.name+'.pth'
            if os.path.exists(path_actor) and os.path.exists(path_critic):
                try:
                    
                    Mod_actor=torch.load(path_actor)
                    Mod_critic=torch.load(path_critic)
                    self.actor.load_state_dict(Mod_actor['model'])
                    self.critic.load_state_dict(Mod_critic['model'])
                    self.actor_optimizer.load_state_dict(Mod_actor['optimizer'])
                    self.critic_optimizer.load_state_dict(Mod_critic['optimizer'])
                    #初始化目标网络参数
                    self.target_actor.load_state_dict(Mod_actor['model'])
                    self.target_critic.load_state_dict(Mod_critic['model'])
    

                    self.epoch=Mod_actor['epoch'] 
                except Exception as e:
                    print(e.args)  #输出异常信息
                    

    def save(self,directory=root+'/Mod/'):
        #创建模型文件名
        filename=''
        #filename='w:'+str(self.w)+'h:'+str(self.h)+'c:'+str(self.channel)
        #存放模型到指定路径
        state = {'model': self.actor.state_dict(), 'optimizer': self.actor_optimizer.state_dict(), 'epoch': self.epoch}
        #torch.save(state, '%s/%s_local.pth' % (directory, filename))
        torch.save(state, '%s/DDPG_actor_%s.pth' % (directory,self.name))
        state = {'model': self.critic.state_dict(), 'optimizer': self.critic_optimizer.state_dict(), 'epoch': self.epoch}
        #torch.save(state, '%s/%s_target.pth' % (directory, filename))
        torch.save(state, '%s/DDPG_critic_%s.pth' % (directory,self.name))


    def learn_on_policy(self,transition_dict):
        #AC算法是进行在线优化，故使用transition_dict
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }

        #进行训练
        # states = torch.tensor(transition_dict['states'],dtype=torch.float).to(device)
        # actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(device)
        # rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)

        #batch = Transition(*zip(*transition_dict))       
        states = torch.cat(transition_dict["states"])
        actions = torch.cat(transition_dict["actions"])
        rewards = torch.cat(transition_dict["rewards"])
        next_states = torch.cat(transition_dict["next_states"])
        dones = torch.cat(transition_dict["dones"])
        #进行离线训练

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))

        #防止梯度过大
        log_probs = torch.clamp(log_probs, -100, 0)

        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        #训练器参数为训练状态
        if self.Is_Train:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 计算策略网络的梯度
            critic_loss.backward()  # 计算价值网络的梯度
            self.actor_optimizer.step()  # 更新策略网络的参数
            self.critic_optimizer.step()  # 更新价值网络的参数
        

        
        #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        train_result['loss']=actor_loss
        return train_result  #返回训练结果

    def learn_off_policy(self):
        #离线训练
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }

        #进行离线训练
        if len(self.replay_memory.memory) < self.Batch_Size:
            return train_result
        transitions = self.replay_memory.sample(self.Batch_Size)  #获取批量经验数据
        batch = Transition(*zip(*transitions))       
        states = torch.cat(batch.state)
        tensor_list = [FloatTensor(arr) for arr in batch.action]
        actions = torch.cat( tensor_list)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        
        #训练器参数为训练状态
        if self.Is_Train:
            next_q_values = self.target_critic(next_states, self.target_actor(next_states))
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
            critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -torch.mean(self.critic(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

        

        #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        train_result['loss']=actor_loss
        return train_result  #返回训练结果
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

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