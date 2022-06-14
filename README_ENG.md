README: [English](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README_ENG.md) | [中文](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README.md)
# DDPG_Ant_v3_Pytorch
## Project background
play OpenAI Gym's Ant-v3 with deep deterministic policy gradient and let ants learn to walk.

## Getting Started Guide

### Install some environments
* To use the gym library to directly call the ant-v3 game environment, you need to install mujoco firstly.If mujoco has not been installed, please see [***Link***](https://zhuanlan.zhihu.com/p/502112539)
* Install OpenAI's gym library
* Install pythorch library
### Test whether the dependent environment can be used normally
* Before starting everything, please make sure that your computer can run mujoco's game environment normally. You can test with the following code first:
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
If the above code can call up the ant game environment, everything is normal.
### Deploy
* The introduction of ant game environment is explained in detail in my code comments.The CPU is used, and the training takes 20-25min. During the test, it can only be said that the ant can run, but it looks like a baby titan.（...Suddenly it became strange ）
### Frame
* pytorch frame
### Code usage
1. Open ddpg_ Ant_ V3.py file, first set the switch flag to 0(As shown in the following figure), train first, and stop training directly after 20-25min training (Don't wait for it all the time)，Because the parameters of the neural network have been saved in the model_DDPG_actor_Ant_v3.pth.

![image](https://user-images.githubusercontent.com/64433060/173536662-31fc127d-372e-415b-8e9e-ddcd5b150031.png)

2. Then, set the switch flag to 1, and you can see the effect of training.

### Version
* I don't think the version information is important, but I still give it for reference. Gym version：0.20.0；pytorch version：1.10.0+cu113；mujoco version：150；mujoco_py version：v1.50.1.0;Finally,I used win10.
### Remark
* The parameters of the neural network are saved in the E disk of the computer. Don't tell me that your computer doesn't have E disk. 

### Contribute
→🤡←

### Acknowledgment
<td align="center"><a href="https://github.com/MorvanZhou"><img src="https://avatars.githubusercontent.com/u/19408436?v=4" width="100px;" alt=""/><br /><sub><b>MorvanZhou</b></sub></a><br /><a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Code">💻</a> <a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Design">🎨</a> <a href="https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow" title="Ideas, Planning, & Feedback">🤔</a></td>

----
## Video demonstration, the effect is as follows:

https://user-images.githubusercontent.com/64433060/168779367-1be1bc87-3591-473e-830e-7b9bc9afba59.mp4



