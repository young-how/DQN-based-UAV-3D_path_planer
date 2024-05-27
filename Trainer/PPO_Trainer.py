#PPO算法训练器
from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
root=os.path.dirname(os.path.abspath(__file__))+'/../'  #根目录
import os
import sys
Path=os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)
 

class PPO_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)   #初始化基类
        actor_param=param.get('actor')
        critic_param=param.get('critic')
        actor_lr=float(actor_param.get('lr'))     #actor学习率
        critic_lr=float(critic_param.get('lr'))  #critic学习率
        

        #actor网络
        self.actor = self.NetworkFactory.Create_Network(actor_param).to(device)  #根据参数创建Q网络
        #critic
        self.critic = self.NetworkFactory.Create_Network(critic_param).to(device)

        #创建名称
        self.name=param.get('name')

        #创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)   #设置优化器，使用adam优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr) 
        PPO_param=param.get('PPO_param')  #获取PPO算法专属的参数值
        self.lmbda=float(PPO_param.get('lmbda'))  
        self.epochs=int(PPO_param.get('epochs'))  #一条序列用来训练的次数
        self.eps=float(PPO_param.get('eps'))    #PPO中断范围参数
        
        
        self.device = device


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
                path_actor=root+'/Mod/actor_PPO.pth'
                path_critic=root+'/Mod/critic_PPO.pth'
            else:
                path_actor=root+'/Mod/actor_PPO_'+self.name+'.pth'
                path_critic=root+'/Mod/critic_PPO_'+self.name+'.pth'
            if os.path.exists(path_actor) and os.path.exists(path_critic):
                try:
                    
                    Mod_actor=torch.load(path_actor)
                    Mod_critic=torch.load(path_critic)
                    self.actor.load_state_dict(Mod_actor['model'])
                    self.critic.load_state_dict(Mod_critic['model'])
                    self.actor_optimizer.load_state_dict(Mod_actor['optimizer'])
                    self.critic_optimizer.load_state_dict(Mod_critic['optimizer'])

                    self.epoch=Mod_actor['epoch'] 
                except Exception as e:
                    print(e.args)  #输出异常信息
                    

    def save(self,directory=root+'/Mod/'):
        #创建模型文件名
        filename=''
        #存放模型到指定路径
        state = {'model': self.actor.state_dict(), 'optimizer': self.actor_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(state, '%s/actor_PPO_%s.pth' % (directory,self.name))
        state = {'model': self.critic.state_dict(), 'optimizer': self.critic_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(state, '%s/critic_PPO_%s.pth' % (directory,self.name))


     # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        if self.IS_Continuous==1:
            #连续动作空间
            next_actions, log_prob = self.actor(next_states)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_states, next_actions)
            q2_value = self.target_critic_2(next_states, next_actions)
            next_value = torch.min(q1_value,
                                q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        else:
            #离散动作空间 
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            q1_value = self.target_critic_1(next_states)
            q2_value = self.target_critic_2(next_states)
            min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),dim=1,keepdim=True)
            next_value = min_qvalue + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +param.data * self.tau)

    def compute_advantage(self,gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
    
    def learn_on_policy(self,transition_dict):
        #PPO算法是进行在线优化，故使用transition_dict
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }

        #训练器参数为训练状态
        if self.Is_Train:
            states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -   dones)
            td_delta = td_target - self.critic(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean( F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        train_result['loss']=actor_loss
        return train_result  #返回训练结果

    def learn_off_policy(self):
        #离线训练
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }
        if self.IS_Continuous==1:
            #连续动作空间训练
            if len(self.replay_memory.memory) < self.Batch_Size:
                return train_result
            transitions = self.replay_memory.sample(self.Batch_Size)  #获取批量经验数据
            batch = Transition(*zip(*transitions))       
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            #rewards=(rewards+100)/100
            
            next_states = torch.cat(batch.next_state)
            dones = torch.cat(batch.done)
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -   dones)
            td_delta = td_target - self.critic(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean( F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数
        train_result['loss']=actor_loss
        return train_result  #返回训练结果
        

    #update更新方法，将输入的数据直接用于更新，适用于onpolicy和offpolicy两类训练方法
    def update(self,transition_dict):
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }
        if transition_dict['states']==[]:
            return train_result  #返回训练结果

        #训练器参数为训练状态
        if self.Is_Train:
            states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -   dones)
            td_delta = td_target - self.critic(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean( F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        

        
        #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数

        train_result['loss']=actor_loss
        return train_result  #返回训练结果
    
    #根据状态选取动作
    def get_action(self,state,eps):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def replace_param(self,target):
        #根据目标网络替换参数
        for target_param, param in zip(target.parameters(), self.actor.parameters()):
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