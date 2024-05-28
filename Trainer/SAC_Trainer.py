#SAC算法训练器
import datetime
import socket
import time

from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
root=os.path.dirname(os.path.abspath(__file__))+'/../'  #根目录
import os
import sys
Path=os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)
 

class SAC_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)   #初始化基类
        actor_param=param.get('actor')
        critic_param=param.get('critic')
        actor_lr=float(actor_param.get('lr'))     #actor学习率
        critic_lr=float(critic_param.get('lr'))  #critic学习率
        self.IsPriority_Replay=int(param.get('IsPriority_Replay'))  #优先级经验回放
        #策略网络
        self.actor = self.NetworkFactory.Create_Network(actor_param).to(device)  #根据参数创建Q网络
        #第一个Q网络
        self.critic_1  = self.NetworkFactory.Create_Network(critic_param).to(device)  #根据参数创建目标网络
        #第二个Q网络
        self.critic_2  = self.NetworkFactory.Create_Network(critic_param).to(device)

        #创建目标网络
        self.target_critic_1  = self.NetworkFactory.Create_Network(critic_param).to(device)
        self.target_critic_2  = self.NetworkFactory.Create_Network(critic_param).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        #创建名称
        self.name=param.get('name')

        #创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)   #设置优化器，使用adam优化器
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr) 
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr) 

        # 使用alpha的log值,可以使训练结果比较稳定
        SAC_param=param.get('SAC_param')  #获取SAC算法专属的参数值
        self.IS_Continuous=int(SAC_param.get('IS_Continuous')) #是否是连续动作
        target_entropy=float(SAC_param.get('target_entropy'))  #target_entropy值
        gamma=float(SAC_param.get('gamma'))  #target_entropy值
        tau=float(SAC_param.get('tau'))  #target_entropy值
        alpha_lr=float(SAC_param.get('alpha_lr'))  #alpha_lr学习率
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.weight_number = 25
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
                path_actor=root+'/Mod/actor_SAC.pth'
                path_critic_1=root+'/Mod/critic_1_SAC.pth'
                path_critic_2=root+'/Mod/critic_2_SAC.pth'
            else:
                path_actor=root+'/Mod/actor_SAC_'+self.name+'.pth'
                path_critic_1=root+'/Mod/critic_1_SAC_'+self.name+'.pth'
                path_critic_2=root+'/Mod/critic_2_SAC_'+self.name+'.pth'
            if os.path.exists(path_actor) and os.path.exists(path_critic_1) and os.path.exists(path_critic_2):
                try:
                    
                    Mod_actor=torch.load(path_actor)
                    Mod_critic_1=torch.load(path_critic_1)
                    Mod_critic_2=torch.load(path_critic_2)
                    self.actor.load_state_dict(Mod_actor['model'])
                    self.critic_1.load_state_dict(Mod_critic_1['model'])
                    self.critic_2.load_state_dict(Mod_critic_2['model'])
                    self.actor_optimizer.load_state_dict(Mod_actor['optimizer'])
                    self.critic_1_optimizer.load_state_dict(Mod_critic_1['optimizer'])
                    self.critic_2_optimizer.load_state_dict(Mod_critic_2['optimizer'])

                    #目标网络复制参数
                    self.target_critic_1.load_state_dict(self.critic_1.state_dict())
                    self.target_critic_2.load_state_dict(self.critic_2.state_dict())

                    self.epoch=Mod_actor['epoch'] 
                except Exception as e:
                    print(e.args)  #输出异常信息


    def save(self,directory=root+'/Mod/'):
        #创建模型文件名
        filename=''
        #存放模型到指定路径
        state = {'model': self.actor.state_dict(), 'optimizer': self.actor_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(state, '%s/actor_SAC_%s.pth' % (directory,self.name))
        state = {'model': self.critic_1.state_dict(), 'optimizer': self.critic_1_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(state, '%s/critic_1_SAC_%s.pth' % (directory,self.name))

        state = {'model': self.critic_2.state_dict(), 'optimizer': self.critic_2_optimizer.state_dict(), 'epoch': self.epoch}
        torch.save(state, '%s/critic_2_SAC_%s.pth' % (directory,self.name))

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
        log_probs = torch.clamp(log_probs, -1, 1)

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
            # 更新两个Q网络
            td_target = self.calc_target(rewards, next_states, dones)
            critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
            critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # 更新策略网络
            new_actions, log_prob = self.actor(states)
            entropy = -log_prob
            q1_value = self.critic_1(states, new_actions)
            q2_value = self.critic_2(states, new_actions)
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新alpha值
            alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)
        else:
            #离散动作空间训练
            #进行离线训练
            if len(self.replay_memory.memory) < self.Batch_Size:
                return train_result
            
            if(self.IsPriority_Replay!=1):
                transitions = self.replay_memory.sample(self.Batch_Size)  #获取批量经验数据
            else:
                transitions,tree_idx,is_weights = self.replay_memory.sample(self.Batch_Size)  #优先级经验回放
            batch = Transition(*zip(*transitions))       
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            next_states = torch.cat(batch.next_state)
            dones = torch.cat(batch.done)


            actor_loss=0
            #训练器参数为训练状态
            if self.Is_Train:
                # 更新两个Q网络
                td_target = self.calc_target(rewards, next_states, dones)
                critic_1_q_values = self.critic_1(states).gather(1, actions)
                critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
                critic_2_q_values = self.critic_2(states).gather(1, actions)
                critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                # 更新策略网络
                probs = self.actor(states)
                log_probs = torch.log(probs + 1e-8)
                # 直接根据概率计算熵
                entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
                q1_value = self.critic_1(states)
                q2_value = self.critic_2(states)
                min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                                    dim=1,
                                    keepdim=True)  # 直接根据概率计算期望
                actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
                #actor_loss = torch.mean(self.log_alpha.exp() * entropy - min_qvalue)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 更新alpha值
                alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)

        

        # #训练次数到达保存周期，保存actor网络和critic网络
         #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数


        train_result['loss']=actor_loss
        return train_result  #返回训练结果

    #通用型更新方式
    def update(self,transition_dict):
        
        #离线训练
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }
        #没有训练数据，返回
        if all(len(sublist) == 0 for sublist in transition_dict['states']):
            return train_result  #返回训练结果
        if self.IS_Continuous==1:
            #连续动作空间训练
            if len(self.replay_memory.memory) < self.Batch_Size:
                return train_result
            #transitions = self.replay_memory.sample(self.Batch_Size)  #获取批量经验数据
            states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
            #actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1, 1).to(self.device)
            actions = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
            if self.IsPriority_Replay==1:
                tree_idx=torch.tensor(transition_dict['idx'],dtype=torch.float).view(-1, 1).to(self.device)
                is_weights =torch.tensor(transition_dict['weights'],dtype=torch.float).view(-1, 1).to(self.device)
            #rewards=(rewards+3)/3

            # 更新两个Q网络
            td_target = self.calc_target(rewards, next_states, dones)
            Q1=self.critic_1(states, actions)
            Q2=self.critic_2(states, actions)
            critic_1_loss = torch.mean(F.mse_loss(Q1, td_target.detach()))
            critic_2_loss = torch.mean(F.mse_loss(Q2, td_target.detach()))
            # 优先级经验回放
            if self.IsPriority_Replay==1:
                critic_1_loss=is_weights*critic_1_loss
                critic_2_loss=is_weights*critic_2_loss
                abs_errors = torch.abs(torch.min(Q1, Q2) - td_target).detach().numpy().squeeze()
                self.replay_memory.batch_update(tree_idx, abs_errors)  # 更新经验的优先级
                
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # 更新策略网络
            new_actions, log_prob = self.actor(states)
            entropy = -log_prob
            q1_value = self.critic_1(states, new_actions)
            q2_value = self.critic_2(states, new_actions)
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新alpha值
            alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)
        else:
            #离散动作空间训练
            #进行离线训练
            if len(self.replay_memory.memory) < self.Batch_Size:
                return train_result
            states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
            rewards=(rewards+10)/10 #奖励重塑


            actor_loss=0
            #训练器参数为训练状态
            if self.Is_Train:
                # 更新两个Q网络
                td_target = self.calc_target(rewards, next_states, dones)
                critic_1_q_values = self.critic_1(states).gather(1, actions.long())
                critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
                critic_2_q_values = self.critic_2(states).gather(1, actions.long())
                critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()
                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                # 更新策略网络
                probs = self.actor(states)
                log_probs = torch.log(probs + 1e-8)
                # 直接根据概率计算熵
                entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
                q1_value = self.critic_1(states)
                q2_value = self.critic_2(states)
                min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                                    dim=1,
                                    keepdim=True)  # 直接根据概率计算期望
                actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
                #actor_loss = torch.mean(self.log_alpha.exp() * entropy - min_qvalue)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 更新alpha值
                alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)
        #训练次数到达保存周期，保存actor网络和critic网络

         #训练次数到达保存周期，保存actor网络和critic网络
        if self.epoch%self.save_loop==0:
            self.save()  #保存网络模型参数


        train_result['loss']=actor_loss
        return train_result  #返回训练结果

     #根据状态选取动作
    def get_action(self,state,eps):
        if self.IS_Continuous==1:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.actor(state)[0]
            return action.tolist()[0]
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return  action.item()
    
    def replace_param(self,target):
        #根据目标网络替换参数
        for target_param, param in zip(target.parameters(), self.actor.parameters()):
            param.data.copy_(target_param.data)

    def Push_Replay(self,Experience,error=0):
        #将经验存放到经验池中
        self.replay_memory.push(Experience,error)

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