README: [English](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README_ENG.md) | [中文](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README.md)
# DDPG_Ant_v3_Pytorch
play OpenAI Gym's Ant-v3 with deep deterministic policy gradient
* To use the gym library to directly call the ant-v3 game environment, you need to install mujoco firstly.If mujoco has not been installed, please see [***Link***](https://zhuanlan.zhihu.com/p/502112539)
* The introduction of ant game environment is explained in detail in my code comments.The CPU is used, and the training takes 20-25min. During the test, it can only be said that the ant can run, but it looks like a baby titan.（...Suddenly it became strange ）
## Code usage：
First set the switch flag to 0, train first, and stop training directly after 20-25min training (Don't wait for it all the time)，Because the parameters of the neural network have been saved in the model_DDPG_actor_Ant_v3.pth.Then, set the switch flag to 1, and you can see the effect of training.
## remark：
1.The parameters of the neural network are saved in the E disk of the computer. Don't tell me that your computer doesn't have E disk. 
2.I don't think the version information is important, but I still give it for reference.
* Gym version：0.20.0；pytorch version：1.10.0+cu113；mujoco version：150；mujoco_py version：v1.50.1.0;Finally,I used win10.

![Ant](https://user-images.githubusercontent.com/64433060/168765453-c92eba63-00c7-40d2-8aa3-b7828b8f9bdd.png)

