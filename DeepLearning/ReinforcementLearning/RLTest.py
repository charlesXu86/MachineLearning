#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: RLTest.py 
@desc: Rl环境测试
@time: 2018/03/28 
"""

import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

env.reset()
random_epsiodes = 0
reward_sum = 0
while random_epsiodes < 10:
    env.render()   # 将图像渲染出来
    observation, reward, done, _ = env.step(np.random.randint(0, 2))  # env.step() 执行随机的action，如果done标记为True，则饰演结束
    reward_sum += reward
    if done:
        random_epsiodes += 1
        print('Reward for this epsidoes was:', reward_sum)
        reward_sum = 0
        env.reset()