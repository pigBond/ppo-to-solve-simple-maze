import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()

        self.maze = np.array(maze)
        self.target = (len(maze)-1, len(maze[0])-1)  # 假设终点是迷宫的右下角
        self.start = (0, 0)  # 假设起点是迷宫的左上角

        # 定义动作和状态空间
        self.action_space = spaces.Discrete(4)  # 四个动作：上(0)、下(1)、左(2)、右(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.maze.shape, dtype=int)

        self.current_position = self.start

    def step(self, action):
        x, y = self.current_position
        done = False  # 初始化 done 为 False

        if action == 0:  # 向上
            x -= 1
        elif action == 1:  # 向下
            x += 1
        elif action == 2:  # 向左
            y -= 1
        elif action == 3:  # 向右
            y += 1

        # 检查是否撞墙或出界
        if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x, y] == 1:
            reward = -5
        elif (x, y) == self.target:
            reward = 20
            done = True  # 明确设置 done 为 True
        else:
            reward = -1
            done = False  # 这里再次确认 done 为 False，确保逻辑清晰

        # 更新智能体的当前位置，确保不越界
        self.current_position = (max(0, min(x, self.maze.shape[0]-1)), max(0, min(y, self.maze.shape[1]-1)))
        return self.current_position, reward, done, {}

    def reset(self):
        # 重置环境状态，智能体回到起点
        self.current_position = self.start
        return self.current_position

    def render(self, mode='human'):
        if mode == 'human':
            plt.imshow(self.maze, cmap='gray')  # 显示迷宫，用灰度图表示墙壁和通路
            plt.scatter(self.current_position[1], self.current_position[0], c='red', s=100)  # 在迷宫中显示小人的位置，红色标记
            plt.pause(0.1)  # 更新画面之前暂停0.1秒，以便观察到小人的移动
            plt.draw()  # 更新画面

