README: [English](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README_ENG.md) | [中文](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README.md)
# DDPG_Ant_v3_Pytorch
## 项目背景
使用深度强化学习算法DDPG玩mujoco里的Ant-3游戏，让蚂蚁学会行走。
## 上手指南
### 安装一些环境
* 想使用gym库直接调用Ant-v3游戏环境，你需要安装mujoco。mujoco还没安装的话，请看[***链接***](https://zhuanlan.zhihu.com/p/502112539)
* 安装OpenAI的gym库
* 安装pytorch库
### 测试依赖环境是否可以正常使用
* 在开始一切之前，请您确保电脑能正常运行mujoco的游戏环境。可以先用以下代码测试：
```
import gym
env = gym.make('Ant-v3')#Hopper-v3\HalfCheetah-v3\Ant-v3\Humanoid-v3
env = env.unwrapped
for episode in range(20):
    observation = env.reset() #环境重置
    while True:
        env.render() #可视化
        action = env.action_space.sample() #动作采样
        observation_, reward, done, info = env.step(action) #单步交互
        if done:
            print('Episode {}'.format(episode))
            break
        observation=observation_
env.close()
```
如果上面的代码可以调出Ant游戏环境，则一切正常。
### 部署
* 关于Ant游戏环境的介绍，在我的代码注释里有详细的解读。使用的是CPU，训练需20-25min，测试时只能说蚂蚁可以跑，但是它是一只奇行种（。。。。。突然间就变得奇怪起来了呢 ）。
### 框架
* 代码基于pytorch框架
### 代码用法
1. 打开DDPG_Ant_v3.py文件，把此文件里面的Switch标志为赋为0（如下图所示），先训练，训练个20-25min就直接停止训练（不要等了，如果让它自然地训练结束会等到猴年马月的），因为神经网络的参数已经被我们保存在了model_DDPG_actor_Ant_v3.pth里。


![image](https://user-images.githubusercontent.com/64433060/173536662-31fc127d-372e-415b-8e9e-ddcd5b150031.png)


2. 然后，再把Switch标志为赋为1，就可以看到训练的效果了。
### 版本信息
* 我感觉版本信息不重要，但还是给一下以供参考。我用的gym版本：0.20.0；我用的pytorch版本：1.10.0+cu113；我用的mujoco应用程序版本：150；我用的mujoco_py包版本：v1.50.1.0
### Remark
* 神经网络的参数被保存在了电脑E盘里，别告诉我你的电脑没有E盘。没有自己改代码。

### 作者
→🤡←

### 鸣谢
<td align="center"><a href="https://github.com/MorvanZhou"><img src="https://avatars.githubusercontent.com/u/19408436?v=4" width="100px;" alt=""/><br /><sub><b>MorvanZhou</b></sub></a><br /><a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Code">💻</a> <a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Design">🎨</a> <a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Ideas, Planning, & Feedback">🤔</a></td>

---
## 视频演示，效果如下：

https://user-images.githubusercontent.com/64433060/168779230-ffd44236-1306-4cda-86e4-b2a9c7db1182.mp4

