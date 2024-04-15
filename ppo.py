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
from actor_critic import ActorCritic
from rollout_buffer import RolloutBuffer

device = setup_device()

class PPO:  # 定义PPO类
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        # 构造函数，初始化PPO对象
        self.has_continuous_action_space = has_continuous_action_space  # 是否有连续的动作空间
        if has_continuous_action_space:
            self.action_std = action_std_init  # 连续动作空间的初始动作标准差
        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # PPO裁剪参数
        self.K_epochs = K_epochs  # 每次更新时的迭代次数
        self.buffer = RolloutBuffer()  # 初始化用于存储经验的缓冲区
        # 初始化策略网络，并将其移动到指定的设备上（例如GPU）
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 设置优化器，分别为actor和critic设置不同的学习率
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        # 初始化旧策略网络，用于在优化时计算策略比率
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 将当前策略的权重复制到旧策略中
        self.MseLoss = nn.MSELoss()  # 初始化均方误差损失，用于计算价值函数的损失

    def set_action_std(self, new_action_std):
        # 设置新的动作标准差，仅适用于连续动作空间
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            # 更新当前策略和旧策略中的动作标准差
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            # 如果在离散动作空间中调用此方法，则打印警告
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 动作标准差的衰减，用于连续动作空间中的探索减少
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate  # 根据衰减率更新动作标准差
            self.action_std = round(self.action_std, 4)  # 保留四位小数
            if (self.action_std <= min_action_std):  # 如果动作标准差低于最小值，则设置为最小值
                self.action_std = min_action_std
            print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)  # 更新策略中的动作标准差
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        # 根据给定的状态选择动作，适用于连续和离散动作空间
        if self.has_continuous_action_space:
            with torch.no_grad():  # 禁用梯度计算
                state = torch.FloatTensor(state).to(device)  # 将状态转换为张量并移动到设备上
                # 使用旧策略网络生成动作、动作的对数概率和状态价值
                action, action_logprob, state_val = self.policy_old.act(state)
                # 将这些值存储到缓冲区中
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                return action.detach().cpu().numpy().flatten()  # 返回处理后的动作
        else:
            with torch.no_grad():  # 禁用梯度计算
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                return action.item()  # 返回单个动作值

    def update(self):
      # 更新策略的核心方法，包括计算回报、优势、损失，并进行梯度下降
      # Monte Carlo estimate of returns
      rewards = []  # 用于存储折扣后的回报
      discounted_reward = 0  # 初始化折扣回报
      for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
          if is_terminal:
              discounted_reward = 0  # 如果是终止状态，则重置折扣回报
          discounted_reward = reward + (self.gamma * discounted_reward)  # 计算折扣回报
          rewards.insert(0, discounted_reward)  # 将折扣回报插入到列表的开头

      # Normalizing the rewards
      rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 将回报列表转换为张量
      rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 归一化处理

      # 将缓冲区中的状态、动作、对数概率和状态值列表转换为张量
      old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
      old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
      old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
      old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

      # 计算优势
      advantages = rewards.detach() - old_state_values.detach()

      # 为K个时期优化策略
      for _ in range(self.K_epochs):
          # 评估旧动作和值
          logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

          # 匹配状态值张量的维度与回报张量
          state_values = torch.squeeze(state_values)

          # 计算比率 (pi_theta / pi_theta__old)
          ratios = torch.exp(logprobs - old_logprobs.detach())

          # 计算替代损失
          surr1 = ratios * advantages
          surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

          # 计算裁剪的目标PPO的最终损失
          loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

          # 进行梯度步骤
          self.optimizer.zero_grad()
          loss.mean().backward()
          self.optimizer.step()

      # 将新权重复制到旧策略
      self.policy_old.load_state_dict(self.policy.state_dict())

      # 清空缓冲区
      self.buffer.clear()

    def save(self, checkpoint_path):
        # 保存模型的方法
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加载模型的方法
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
