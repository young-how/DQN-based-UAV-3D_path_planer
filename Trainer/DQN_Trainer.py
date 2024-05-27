#DQN算法训练器
from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
#sys.path.append("E:\younghow\RLGF")
#sys.path.append("E:\younghow\RLGF\Envs")
#sys.path.append("E:\younghow\RLGF\BaseClass")
root=os.path.dirname(os.path.abspath(__file__))+'/../'  #根目录
import os
import sys
Path=os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练

class DQN_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)   #初始化基类

        
        # self.q_local = QNetwork(h=self.h, w=self.w, outputs=self.output).to(device)   #初始化Q网络
        # self.q_target = QNetwork(h=self.h, w=self.w, outputs=self.output).to(device) 
        # self.state_dime=int(param.get('state_dime'))
        # self.action_dim =int(param.get('action_dime'))
        # self.q_local = Qnet(4,128,2).to(device)   #初始化Q网络
        # self.q_target = Qnet(4,128,2).to(device) 
        # self.q_local = Qnet(10,128,8).to(device)   #初始化Q网络
        # self.q_target = Qnet(10,128,8).to(device) 
        self.q_local = self.NetworkFactory.Create_Network(param).to(device)  #根据参数创建Q网络
        self.q_target = self.NetworkFactory.Create_Network(param).to(device)  #根据参数创建目标网络

        #创建优化器
        self.optim = optim.Adam(self.q_local.parameters(), lr=self.LEARNING_RATE)   #设置优化器，使用adam优化器
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
            self.q_target.load_state_dict(Mod['model'])
            self.q_local.load_state_dict(Mod['model'])
            self.optim.load_state_dict(Mod['optimizer'])
            self.epoch=Mod['epoch'] 
        else:
            if self.name!=None:
                path_Qtarget=root+'/Mod/q_target_'+self.name+'.pth'
                path_Qlocal=root+'/Mod/q_local_'+self.name+'.pth'
            else:
                path_Qtarget=root+'/Mod/q_target.pth'
                path_Qlocal=root+'/Mod/q_local.pth'
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
        torch.save(state, '%s/q_local_%s.pth' % (directory) %(self.name))
        state = {'model': self.q_local.state_dict(), 'optimizer': self.optim.state_dict(), 'epoch': self.epoch}
        #torch.save(state, '%s/%s_target.pth' % (directory, filename))
        torch.save(state, '%s/q_target_%s.pth' % (directory) %(self.name))


    def learn_off_policy(self):
        self.epoch+=1
        train_result={'sum_epoch':self.epoch,'loss':self.loss }

        #进行训练
        if len(self.replay_memory.memory) < self.Batch_Size:
            return train_result
            
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

        Q_targets_next = self.q_target(next_states).detach().max(1)[0].reshape(Q_expected.shape) #计算Q目标值估计 #注：源代码写法
        #Q_targets_next = self.q_target(next_states).gather(1, actions) 
        #Q_targets_next = self.q_target(next_states).gather(1, actions)    #计算Q目标值估计 #注：修改版写法

        # Compute the expected Q values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))   #更新Q目标值
        #训练Q网络
        #self.q_local.train(mode=True)        
        
        #loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))  #计算误差
        loss = self.mse_loss(Q_expected, Q_targets)  #计算误差
        #训练器参数为训练状态
        if self.Is_Train:
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

    def hard_update(self):
        #更新目标网络
        for target_param, param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(param.data)
 
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