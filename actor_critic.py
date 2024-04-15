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

from utils import setup_device

device = setup_device()

class ActorCritic(nn.Module):  # 定义ActorCritic类，继承自nn.Module
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()  # 调用父类的构造函数
        self.has_continuous_action_space = has_continuous_action_space  # 标记是否为连续动作空间

        if has_continuous_action_space:  # 如果是连续动作空间
            self.action_dim = action_dim  # 设置动作维度
            # 初始化动作的方差，存储在action_var变量中
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor部分
        if has_continuous_action_space:  # 如果是连续动作空间
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),  # 输入层到隐藏层，64个神经元
                nn.Tanh(),  # 激活函数
                nn.Linear(64, 64),  # 隐藏层到隐藏层，64个神经元
                nn.Tanh(),  # 激活函数
                nn.Linear(64, action_dim),  # 隐藏层到输出层
                nn.Tanh()  # 输出层激活函数，用于输出动作的均值
            )
        else:  # 如果是离散动作空间
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),  # 输入层到隐藏层
                nn.Tanh(),  # 激活函数
                nn.Linear(64, 64),  # 隐藏层到隐藏层
                nn.Tanh(),  # 激活函数
                nn.Linear(64, action_dim),  # 隐藏层到输出层
                nn.Softmax(dim=-1)  # 输出层激活函数，用于输出动作的概率分布
            )

        # critic部分
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层到隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 64),  # 隐藏层到隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(64, 1)  # 隐藏层到输出层，输出一个值作为状态价值估计
        )

    def set_action_std(self, new_action_std):  # 设置新的动作标准差
        if self.has_continuous_action_space:  # 如果是连续动作空间
            # 更新动作的方差
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:  # 如果不是连续动作空间，打印警告信息
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):  # 定义前向传播函数，但未实现
        raise NotImplementedError

    def act(self, state):  # 根据状态生成动作、动作的对数概率和状态价值
        if self.has_continuous_action_space:  # 如果是连续动作空间
            action_mean = self.actor(state)  # 计算动作的均值
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 计算协方差矩阵
            dist = MultivariateNormal(action_mean, cov_mat)  # 定义多变量正态分布
        else:  # 如果是离散动作空间
            action_probs = self.actor(state)  # 计算动作的概率分布
            dist = Categorical(action_probs)  # 定义分类分布

        action = dist.sample()  # 从分布中采样一个动作
        action_logprob = dist.log_prob(action)  # 计算采样动作的对数概率
        state_val = self.critic(state)  # 计算状态价值

        return action.detach(), action_logprob.detach(), state_val.detach()  # 返回动作、对数概率和状态价值

    def evaluate(self, state, action):  # 评估给定状态和动作
        if self.has_continuous_action_space:  # 如果是连续动作空间
            action_mean = self.actor(state)  # 计算动作的均值
            action_var = self.action_var.expand_as(action_mean)  # 扩展动作方差的维度与均值相同
            cov_mat = torch.diag_embed(action_var).to(device)  # 构建对角协方差矩阵
            dist = MultivariateNormal(action_mean, cov_mat)  # 定义多变量正态分布
            # 如果动作维度为1，调整动作的形状以匹配期望的维度
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:  # 如果是离散动作空间
            action_probs = self.actor(state)  # 计算动作的概率分布
            dist = Categorical(action_probs)  # 定义分类分布

        action_logprobs = dist.log_prob(action)  # 计算给定动作的对数概率
        dist_entropy = dist.entropy()  # 计算分布的熵
        state_values = self.critic(state)  # 计算状态价值

        return action_logprobs, state_values, dist_entropy  # 返回对数概率、状态价值和分布的熵
