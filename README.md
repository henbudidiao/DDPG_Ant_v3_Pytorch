README: English | [中文](https://github.com/henbudidiao/DDPG_Ant_v3_Pytorch/blob/main/README.md)
# DDPG_Ant_v3_Pytorch
play OpenAI Gym's Ant-v3 with deep deterministic policy gradient
* 想使用gym库直接调用Ant-v3游戏环境，你需要安装mujoco。mujoco还没安装的话，请看[***链接***](https://zhuanlan.zhihu.com/p/502112539)
* 关于Ant游戏环境的介绍，在我的代码注释里有详细的解读。使用的是CPU，训练需15-20min，测试时只能说蚂蚁可以跑，但是它是一只奇行种（。。。。。突然间就变得奇怪起来了呢 ）。
## 代码用法：
先把Switch标志为赋为0，先训练，训练个20-25min就直接停止训练（不要等了，如果让它自然地训练结束会等到猴年马月的），因为神经网络的参数已经被我们保存在了model_DDPG_actor_Ant_v3.pth里。然后，把Switch标志为赋为1，就可以看到训练的效果了。
## remark：
1.神经网络的参数被保存在了电脑E盘里，别告诉我你的电脑没有E盘。没有自己改代码。
2.我感觉版本信息不重要，但还是给一下以供参考。我用的gym版本：0.20.0；我用的pytorch版本：1.10.0+cu113；我用的mujoco应用程序版本：150；我用的mujoco_py包版本：v1.50.1.0

![Ant](https://user-images.githubusercontent.com/64433060/168765453-c92eba63-00c7-40d2-8aa3-b7828b8f9bdd.png)
