# DDPG_Ant_v3_Pytorch
play mujoco Ant-v3 with DDPG
想使用gym库直接调用Ant-v3游戏环境，你需要安装mujoco。mujoco还没安装的话，请看***链接***
关于Ant游戏环境的介绍，在我的代码注释里有详细的解读。使用的是CPU，训练需15-20min，测试时只能说蚂蚁可以跑，但是它是一只奇行种（。。。。。突然间就变得奇怪起来了呢 ）。
# 代码用法：

先把Switch标志为赋为0，先训练，训练个20-25min就直接停止训练（不要等了，如果让它自然地训练结束会等到猴年马月的），因为神经网络的参数已经被我们保存在了model_DDPG_actor_Ant_v3.pth里。然后，把Switch标志为赋为1，就可以看到训练的效果了。
![QQ截图20220517114230](https://user-images.githubusercontent.com/64433060/168765453-c92eba63-00c7-40d2-8aa3-b7828b8f9bdd.png)
