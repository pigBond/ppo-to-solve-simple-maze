import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

import gym
# import roboschool
import pybullet_envs

from ppo import PPO

# 打印分隔线，标示训练开始的部分
print("============================================================================================")
# 训练部分的开始
################################### Training ###################################
# 初始化环境的超参数
# env_name = "CartPole-v1"  # 环境名称
# has_continuous_action_space = False  # 标记动作空间是否为连续
# max_ep_len = 400  # 一个episode的最大时间步数

env_name = "LunarLander-v2"
has_continuous_action_space = False
max_ep_len = 300
action_std = None


# env_name = "BipedalWalker-v2"
# has_continuous_action_space = True
# max_ep_len = 1500           # max timesteps in one episode
# action_std = 0.1            # set same std for action distribution which was used while saving


# env_name = "RoboschoolWalker2d-v1"
# has_continuous_action_space = True
# max_ep_len = 1000           # max timesteps in one episode
# action_std = 0.1            # set same std for action distribution which was used while saving


max_training_timesteps = int(1e5)  # 训练的最大时间步数
print_freq = max_ep_len * 4  # 打印平均奖励的频率（以时间步为单位）
log_freq = max_ep_len * 2  # 日志记录平均奖励的频率（以时间步为单位）
save_model_freq = int(2e4)  # 保存模型的频率（以时间步为单位）
action_std = None  # 初始的动作标准差，适用于连续动作空间
#####################################################
# 注释说明打印和日志记录的频率应大于max_ep_len
# PPO算法的超参数设置
update_timestep = max_ep_len * 4  # 更新策略的时间步频率
K_epochs = 40  # 每次更新策略时的训练轮数
eps_clip = 0.2  # PPO算法中的裁剪参数
gamma = 0.99  # 折扣因子
lr_actor = 0.0003  # actor网络的学习率
lr_critic = 0.001  # critic网络的学习率
random_seed = 0  # 随机种子设置（0表示不使用随机种子）
#####################################################
# 打印当前训练的环境名称
print("training environment name : " + env_name)
env = gym.make(env_name)  # 创建环境实例
# 获取状态空间的维度
state_dim = env.observation_space.shape[0]
# 根据动作空间类型获取动作维度
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n
###################### logging ######################
# 设置日志目录，不覆盖多次运行的日志文件
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 获取日志目录中文件的数量，用于生成新的日志文件名
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)
# 创建新的日志文件
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################
################### checkpointing ###################
# 设置预训练模型的运行编号，以防覆盖同一环境下的权重
run_num_pretrained = 0
directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
# 定义检查点路径
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#####################################################
# 打印所有超参数
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("--------------------------------------------------------------------------------------------")
# 根据动作空间类型打印不同的初始化信息
if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    # 这里缺少关于动作标准差衰减的具体代码，但正常情况下应包括衰减率、最小标准差等信息
else:
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    # 设置环境、PyTorch和Numpy的随机种子，确保实验的可重复性
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
#####################################################
# 再次打印分隔线，表示训练设置完成
print("============================================================================================")


################# training procedure ################
################# 训练程序 ################

# 初始化一个PPO代理
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

# 跟踪总训练时间
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")

# 日志文件
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# 打印和记录变量
print_running_reward = 0
print_running_episodes = 0
log_running_reward = 0
log_running_episodes = 0
time_step = 0
i_episode = 0

# 训练循环
while time_step <= max_training_timesteps:
    state = env.reset()
    current_ep_reward = 0
    for t in range(1, max_ep_len+1):
        # 使用策略选择动作
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)

        #env.render()  # 在屏幕上渲染环境状态

        # 保存奖励和是否终止
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward

        # 更新PPO代理
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # 如果是连续动作空间；则根据频率衰减动作标准差
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # 在日志文件中记录
        if time_step % log_freq == 0:
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_running_reward = 0
            log_running_episodes = 0

        # 打印平均奖励
        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            print_running_reward = 0
            print_running_episodes = 0

        # 保存模型权重
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        # 如果本轮结束，则跳出循环
        if done:
            break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1

    i_episode += 1

log_f.close()
env.close()

# 打印总训练时间
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")