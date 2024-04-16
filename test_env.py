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
        self.step_count = 0  # 步骤计数器

    def step(self, action):
        x, y = self.current_position
        done = False  # 初始化 done 为 False
        # 预测新位置
        if action == 0:  # 向上
            x -= 1
        elif action == 1:  # 向下
            x += 1
        elif action == 2:  # 向左
            y -= 1
        elif action == 3:  # 向右
            y += 1
        
        # 检查新位置是否合法
        if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x, y] == 1:
            reward = -5
            x, y = self.current_position  # 保持在原位置
        elif (x, y) == self.target:
            reward = 20
            done = True  # 明确设置 done 为 True
        else:
            reward = -1

        self.current_position = (x, y)
        self.step_count += 1  # 更新步骤计数
        return self.current_position, reward, done, {}

    def reset(self):
        self.current_position = self.start
        self.step_count = 0  # 重置步骤计数器
        return self.current_position

    def render(self, mode='human'):
        if mode == 'human':
            plt.imshow(self.maze, cmap='gray')  # 显示迷宫
            plt.scatter(self.current_position[1], self.current_position[0], c='red', s=100)  # 标记当前位置
            plt.text(self.current_position[1], self.current_position[0], str(self.step_count), color='blue', fontsize=12, ha='center')  # 标注步数
            plt.pause(0.1)  # 更新画面之前暂停
            plt.draw()  # 更新画面

# 创建迷宫和环境进行测试
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

env = MazeEnv(maze)
env.reset()
plt.ion()  # 开启交互模式
done = False
while not done:
    action = env.action_space.sample()  # 随机选择一个动作
    state, reward, done, _ = env.step(action)
    env.render()
plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图像
