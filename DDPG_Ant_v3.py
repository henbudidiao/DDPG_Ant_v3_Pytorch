#开发者：Bright Fang
#开发时间：2022/5/15 9:35
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
env = gym.make('Ant-v3').unwrapped#Hopper-v3\HalfCheetah-v3\Ant-v3\Humanoid-v3
'''Ant环境：状态有111个，前27个有值，包括蚂蚁身体不同部位的位置值，接着是这些单独部位的速度(它们的导数)
，后86=14*6个是施加到每个连杆质心的接触力，这14个环节是:地面环节、躯干环节、每条腿的3个环节(1 + 1 + 12)
和6个外力，在v3版本和v2版本接触力都为0。
前27个状态介绍：
    |0|躯干的z坐标|-inf,inf|单位：m                     |1|躯干的x方向|-inf,inf|单位：rad
    |2|躯干的y方向|-inf,inf|单位：rad                   |3|躯干的z方向|-inf,inf|单位：rad
    |4|w-躯干方向 |-inf,inf|单位：rad                   |5|躯干和左前方第一个连杆之间的角度|-inf,inf|单位：rad
    |6|左前方两个连杆之间的角度|-inf,inf|单位：rad       |7|躯干和右前侧第一个连杆之间的角度|-inf,inf|单位：rad
    |8|右前侧两个连杆之间的角度|-inf,inf|单位：rad       |9|躯干和左后侧第一个链接之间的角度|-inf,inf|单位：rad
    |10|左后侧两个链接之间的角度|-inf,inf|单位：rad      |11|躯干和右后侧第一个环节之间的角度|-inf,inf|单位：rad
    |12|右后侧两个链接之间的角度|-inf,inf|单位：rad      |13|躯干的x坐标速度|-inf,inf|单位：m/s
    |14|躯干的y坐标速度|-inf,inf|单位：m/s               |15|躯干的z坐标速度|-inf,inf|单位：m/s
    |16|躯干的x坐标角速度|-inf,inf|单位：rad/s           |17|躯干的y坐标角速度|-inf,inf|单位：rad/s
    |18|躯干的z坐标角速度|-inf,inf|单位：rad/s           |19|躯干和左前连杆之间角度的角速度|-inf,inf|单位：rad 
    |20|左前连杆之间角度的角速度|-inf,inf|单位：rad      |21|躯干和右前连杆之间角度的角速度|-inf,inf|单位：rad 
    |22|右前连杆之间角度的角速度|-inf,inf|单位：rad      |23|躯干和左后连杆之间角度的角速度|-inf,inf|单位：rad 
    |24|左后连杆之间角度的角速度|-inf,inf|单位：rad      |25|躯干和右后连杆之间角度的角速度|-inf,inf|单位：rad 
    |26|右后连杆之间角度的角速度|-inf,inf|单位：rad   
动作是8个施加在铰链接合处的扭矩，是连续量。
8个动作介绍：
    |0|施加在躯干和左前臀部之间的转子上的扭矩|-1,1|单位：N*m   |1|施加在左前两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |2|施加在躯干和右前臀部之间的转子上的扭矩|-1,1|单位：N*m   |9|施加在右前两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |4|施加在躯干和左后臀部之间的转子上的扭矩|-1,1|单位：N*m   |5|施加在左后两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |6|施加在躯干和右后臀部之间的转子上的扭矩|-1,1|单位：N*m   |7|施加在右后两个连杆之间的转子上的扭矩|-1,1|单位：N*m
4个奖励介绍：
奖励由四个部分组成：reward=healthy_r+forward_r-contrl_cost-contact_cost
healthy_r:每个时间里蚂蚁是否健康，每健康一秒它得到一个固定值的奖励
forward_r：前进的奖励，如果蚂蚁向前移动(在正x方向)，这个奖励将是正的。
control_cost:如果蚂蚁采取过大的行动，惩罚蚂蚁的负奖励
contact_cost:如果外部接触力过大，惩罚蚂蚁的负奖励  
2个结束介绍：
游戏的回合结束介绍——done：      
1.如果蚂蚁不健康就结束回合（环境自带）
2.如果持续时间达到500个时间步长就结束（自己给的）
        '''
state_number=env.observation_space.shape[0]
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
LR_A = 5e-5    # learning rate for actor
LR_C = 1e-4    # learning rate for critic
GAMMA = 0.95
# reward discount
MemoryCapacity=10000
Batch=128
Switch=0
tau = 0.0005
RENDER = False
'''DDPG第一步 设计A-C框架的Actor（DDPG算法，只有critic的部分才会用到记忆库）'''
'''第一步 设计A-C框架形式的网络部分'''
class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,512)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(512,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        act=max_action*torch.tanh(self.out(inputstate))#对输出的动作力矩限幅在[-1,1]
        return act
class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        self.in_to_y1=nn.Linear(input,512)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(512,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,output)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        act=self.out(inputstate)
        return act
class Actor():
    def __init__(self):
        self.actor_estimate_eval,self.actor_reality_target = ActorNet(state_number,action_number),ActorNet(state_number,action_number)
        self.optimizer = torch.optim.Adam(self.actor_estimate_eval.parameters(), lr=LR_A)

    '''第二步 编写根据状态选择动作的函数'''

    def choose_action(self, s):
        inputstate = torch.FloatTensor(s)
        probs = self.actor_estimate_eval(inputstate)
        return probs

    '''第四步 编写A的学习函数'''
    '''生成输入为s的actor估计网络，用于传给critic估计网络，虽然这与choose_action函数一样，但如果直接用choose_action
    函数生成的动作，DDPG是不会收敛的，原因在于choose_action函数生成的动作经过了记忆库，动作从记忆库出来后，动作的梯度数据消失了
    所以再次编写了learn_a函数，它生成的动作没有过记忆库，是带有梯度的'''

    def learn_a(self, s):
        s = torch.FloatTensor(s)
        A_prob = self.actor_estimate_eval(s)
        return A_prob

    '''把s_输入给actor现实网络，生成a_，a_将会被传给critic的实现网络'''

    def learn_a_(self, s_):
        s_ = torch.FloatTensor(s_)
        A_prob = self.actor_reality_target(s_).detach()  # 这个东西要送给critic网络
        return A_prob

    '''actor的学习函数接受来自critic估计网络算出的Q_estimate_eval当做自己的loss，即负的critic_estimate_eval(s,a)，使loss
    最小化，即最大化critic网络生成的价值'''

    def learn(self, a_loss):
        loss = a_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''第六步，最后一步  编写软更新程序，Actor部分与critic部分都会有软更新代码'''
    '''DQN是硬更新，即固定时间更新，而DDPG采用软更新，w_老_现实=τ*w_新_估计+(1-τ)w_老_现实'''
    def soft_update(self):
        for target_param, param in zip(self.actor_reality_target.parameters(), self.actor_estimate_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Critic():
    def __init__(self):
        self.critic_estimate_eval,self.critic_reality_target=CriticNet(state_number+action_number,action_number),CriticNet(state_number+action_number,action_number)
        self.optimizer = torch.optim.Adam(self.critic_estimate_eval.parameters(), lr=LR_C)
        self.lossfun=nn.MSELoss()

    '''第五步 编写critic的学习函数'''
    '''使用critic估计网络得到 actor的loss，这里的输入参数a是带梯度的'''

    def learn_loss(self, s, a):
        s = torch.FloatTensor(s)
        # a = a.view(-1, 1)
        Q_estimate_eval = -self.critic_estimate_eval(s, a).mean()
        return Q_estimate_eval

    '''这里的输入参数a与a_是来自记忆库的，不带梯度，根据公式我们会得到critic的loss'''

    def learn(self, s, a, r, s_, a_):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)#当前动作a来自记忆库
        r = torch.FloatTensor(r)
        s_ = torch.FloatTensor(s_)
        # a_ = a_.view(-1, 1)  # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        Q_estimate_eval = self.critic_estimate_eval(s, a)
        Q_next = self.critic_reality_target(s_, a_).detach()
        Q_reality_target = r + GAMMA * Q_next
        loss = self.lossfun(Q_estimate_eval, Q_reality_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.critic_reality_target.parameters(), self.critic_estimate_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

'''第二步  建立记忆库'''
class Memory():
    def __init__(self,capacity,dims):
        self.capacity=capacity
        self.mem=np.zeros((capacity,dims))
        self.memory_counter=0
    '''存储记忆'''
    def store_transition(self,s,a,r,s_):
        tran = np.hstack((s, a,r, s_))  # 把s,a,r,s_困在一起，水平拼接
        index = self.memory_counter % self.capacity#除余得索引
        self.mem[index, :] = tran  # 给索引存值，第index行所有列都为其中一次的s,a,r,s_；mem会是一个capacity行，（s+a+r+s_）列的数组
        self.memory_counter+=1
    '''随机从记忆库里抽取'''
    def sample(self,n):
        assert self.memory_counter>=self.capacity,'记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)#从capacity个记忆里随机抽取n个为一批，可得到抽样后的索引号
        new_mem = self.mem[sample_index, :]#由抽样得到的索引号在所有的capacity个记忆中  得到记忆s，a，r，s_
        return new_mem
'''训练'''
if Switch==0:
    print('DDPG训练中...')
    actor=Actor()
    critic=Critic()
    M=Memory(MemoryCapacity,2 * state_number + action_number + 1)#奖惩是一个浮点数
    for episode in range(2000):
        observation = env.reset() #环境重置
        reward_totle=0
        for timestep in range(500):
            # print("\rtimestep: {}".format(timestep), end="  ")
        # while True:
            if RENDER:
                env.render()
            action=actor.choose_action(observation).detach().numpy()
            # action = env.action_space.sample() #动作采样
            observation_, reward, done, info = env.step(action) #单步交互
            M.store_transition(observation,action,reward,observation_)
            #记忆库存储
            #有的2000个存储数据就开始学习
            if M.memory_counter>MemoryCapacity:
                b_M = M.sample(Batch)
                b_s = b_M[:, :state_number]
                b_a = b_M[:, state_number: state_number + action_number]
                b_r = b_M[:, -state_number - 1: -state_number]
                b_s_ = b_M[:, -state_number:]
                actor_action = actor.learn_a(b_s)#8*100
                actor_action_ = actor.learn_a_(b_s_)
                critic.learn(b_s, b_a, b_r, b_s_, actor_action_)
                Q_c_to_a_loss = critic.learn_loss(b_s, actor_action)
                actor.learn(Q_c_to_a_loss)
                # 软更新
                actor.soft_update()
                critic.soft_update()
            if done :#or timestep>=1000:
                break
            observation = observation_
            reward_totle += reward
        print('Episode {}，奖励：{}'.format(episode, reward_totle))
        if episode % 10 == 0 and episode > 100:#保存神经网络参数
            save_data = {'net': actor.actor_estimate_eval.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': episode}
            torch.save(save_data, "E:\model_DDPG_actor_Ant_v3.pth")
            save_data = {'net': critic.critic_estimate_eval.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': episode}
            torch.save(save_data, "E:\model_DDPG_critic_Ant_v3.pth")
    #     if reward_totle > 800: RENDER = True
    # env.close()
    '''测试'''
else:
    print('DDPG测试中...')
    aa=Actor()
    cc=Critic()
    checkpoint_aa = torch.load("E:\model_DDPG_actor_Ant_v3.pth")
    aa.actor_estimate_eval.load_state_dict(checkpoint_aa['net'])
    checkpoint_cc = torch.load("E:\model_DDPG_critic_Ant_v3.pth")
    cc.critic_estimate_eval.load_state_dict(checkpoint_cc['net'])
    for j in range(10):
        state = env.reset()
        total_rewards = 0
        for timestep in range(500):
            env.render()
            action=aa.choose_action(state).detach().numpy()
            new_state, reward, done, info = env.step(action)  # 执行动作
            total_rewards += reward
            state = new_state
        print("Score", total_rewards)
    env.close()
