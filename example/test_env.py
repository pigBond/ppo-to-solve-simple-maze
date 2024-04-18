import pygame
import numpy as np
from gym import spaces
import gym
from ppo.ppo import PPO

directions = ["上", "下", "左", "右"]


class MazeEnv(gym.Env):
    def __init__(self, maze, n=10, mode="single",render_mode="train", screen_size=400):
        super(MazeEnv, self).__init__()
        self.n = n  # 迷宫的大小 (n x n)
        self.maze = np.array(maze)
        self.reward_array = np.zeros_like(self.maze)  # 创建一个与maze大小相同的数组来存储动态奖励值
        # 路的初始奖励为0
        self.reward_array[self.maze == 2] = 1  # 起点奖励为-1
        self.reward_array[self.maze == 3] = 20 * n  # 终点奖励为-20*n
        self.reward_array[self.maze == 1] = 5  # 墙奖励为-5
        
        self.screen_size = screen_size  # 屏幕尺寸
        self.cell_size = screen_size // n  # 每个单元格的像素大小
        self.action_space = spaces.Discrete(
            4
        )  # 动作空间：上（0），下（1），左（2），右（3）
        self.observation_space = spaces.Box(
            low=0, high=n - 1, shape=(2,), dtype=np.int32
        )  # 观测空间为位置坐标

        # self.current_position == self.state
        self.state = np.array([0, 0])  # 初始化状态为左上角
        self.goal = np.array([n - 1, n - 1])  # 目标位置为右下角

        self.step_count = 0
        self.block_size = 40  # 设置每个格子的大小
        self.mode = mode  # 'single' or 'path'
        self.visited = set()  # 用于存储访问过的位置
        self.effective_movement = False  # 用于存储本次动作是否能产生有效的移动
        self.render_mode=render_mode
        self.init_screen()

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.maze.shape[1] * self.block_size, self.maze.shape[0] * self.block_size)
        )
        pygame.display.set_caption("Maze Visualization")
        self.font = pygame.font.Font(None, 36)  # 字体初始化
        self.render_static()  # 首次渲染静态元素

    def render_static(self):
        """渲染不会变的部分，如墙和路径"""
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                rect = pygame.Rect(
                    y * self.block_size,
                    x * self.block_size,
                    self.block_size,
                    self.block_size,
                )
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 墙体是黑色
                elif self.maze[x, y] == 2:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # 起点是蓝色
                elif self.maze[x, y] == 3:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # 终点是绿色
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # 路是白色

    def reset(self):
        self.state = np.array([0, 0])  # 每次开始重置到左上角
        self.step_count = 0
        self.visited = set([tuple(self.state)])  # 重置访问位置记录
        self.reward_array = np.zeros_like(self.maze)  # 重置奖励数组
        self.reward_array[self.maze == 2] = 1  # 起点奖励为-1
        self.reward_array[self.maze == 3] = 20 * self.n  # 终点奖励为-20*n
        self.reward_array[self.maze == 1] = 5  # 墙奖励为-5
        if self.render_mode=="show":
            self.render_static()  # 重新渲染静态元素
        return self.state

    def is_within_bounds(self, x, y):
        """检查给定位置是否在迷宫范围内"""
        return 0 <= x < self.n and 0 <= y < self.n

    def step(self, action):
        x, y = self.state
        # 根据动作更新状态
        if action == 0:  # 上
            x -= 1
            # self.effective_movement = True
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1
        else:
            self.effective_movement = False
            return self.state, -10000, False, {}  # 无效动作返回特别大的负奖励

        # 检查是否越界
        if not self.is_within_bounds(x, y):
            self.effective_movement = False
            return self.state, -10000, False, {}  # 越界返回特别大的负奖励
        else:
            self.effective_movement = True
            
        if self.maze[x, y] == 1 or self.effective_movement==False:
            # 撞墙,保持在原位置
            # return self.current_position, reward, done
            self.reward_array[x, y] += 10  # 更新撞墙位置的奖励值
            return self.state, -self.reward_array[x, y], False, {}
        elif (x, y) == tuple(self.goal) and self.maze[x, y] == 3:
            # 到达终点
            print(
                "**********************************************************************"
            )
            print("到达终点!")
            print(
                "**********************************************************************"
            )
            return (x, y), 20 * self.n, True, {}
        else:
            # 前面是路,正常前进
            self.state = np.array([x, y])
            self.visited.add(tuple(self.state))
            self.step_count += 1

            # if tuple(self.state) in self.visited:
            #     reward = -2  # 访问已探索位置
            # else:
            #     reward = -1  # 探索新位置

            self.reward_array[x, y] += 1  # 更新访问位置的奖励值

            if self.effective_movement is True:
                # print(
                #     "====================================================================="
                # )
                # print("本次动作有效")
                # print("动作: ", directions[action])
                # print("得到位置: ", self.state)
                # print("步数: ", self.step_count)
                # print(
                #     "====================================================================="
                # )
                self.effective_movement = False
            return self.state, -self.reward_array[x, y], False, {}

    def render(self):
        if self.mode == "single":
            self.render_static()  # 重新渲染静态元素，覆盖旧路径
        # 绘制访问过的路径（如果在路径模式下）
        if self.mode == "path":
            for pos in self.visited:
                rect = pygame.Rect(
                    pos[1] * self.block_size,
                    pos[0] * self.block_size,
                    self.block_size,
                    self.block_size,
                )
                pygame.draw.rect(self.screen, (128, 128, 128), rect)

        # 绘制当前位置和步数
        x, y = self.state
        rect = pygame.Rect(
            y * self.block_size, x * self.block_size, self.block_size, self.block_size
        )
        pygame.draw.rect(self.screen, (128, 128, 128), rect)  # 当前位置用灰色标记
        text = self.font.render(str(self.step_count), True, (0, 0, 0))
        # self.font.render 方法用于创建包含文本的图像
        text_rect = text.get_rect(
            center=(
                y * self.block_size + self.block_size // 2,
                x * self.block_size + self.block_size // 2,
            )
        )
        # text.get_rect() 方法获取文本图像的矩形区域，用于定位。
        # center= 设置文本图像的中心点。计算方法是当前块的中心位置，确保文字居中于矩形。
        self.screen.blit(text, text_rect)

        pygame.display.flip()  # 更新整个屏幕

    def close(self):
        pygame.quit()


# if __name__ == "__main__":

#     # 环境选择和初始化
#     maze = [
#         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
#         [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
#         [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
#         [1, 1, 1, 1, 0, 1, 0, 0, 0, 3],
#     ]
#     # maze = [
#     #     [2, 1, 0, 0, 0],
#     #     [0, 1, 0, 1, 0],
#     #     [0, 1, 0, 1, 0],
#     #     [0, 1, 1, 1, 0],
#     #     [0, 0, 0, 0, 3],
#     # ]

#     # env = gym.make(env_name)
#     env = MazeEnv(maze,n=10, mode="single",render_mode="train")

#     state = env.reset()

#     # 获取环境状态维度和动作维度
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     print("state_dim = ", state_dim)
#     print("action_dim = ", action_dim)

#     # 初始化PPO
#     ppo_agent = PPO(
#         state_dim,
#         action_dim,
#         lr_actor=0.0003,
#         lr_critic=0.001,
#         gamma=0.99,
#         K_epochs=80,
#         eps_clip=0.2,
#         has_continuous_action_space=False,
#     )

#     # 训练参数
#     n_episodes = 500  # 训练周期数
#     max_timesteps = 1500  # 每个周期的最大时间步数
#     log_interval = 10  # 日志记录间隔

#     # 训练循环
#     for episode in range(1, n_episodes + 1):
#         state = env.reset()
#         for t in range(max_timesteps):
#             action = ppo_agent.select_action(state)
#             # print("action = ",action)
#             # env.render()
#             # pygame.time.wait(100)  # 每个动作间隔500毫秒
#             state, reward, done, _ = env.step(action)
#             ppo_agent.buffer.rewards.append(reward)
#             ppo_agent.buffer.is_terminals.append(done)

#             # print(env.reward_array)

#             if done:
#                 break
#         ppo_agent.update()  # 更新策略

#         if episode % log_interval == 0:
#             print(f"Episode {episode}/{n_episodes} completed")

#     # 保存模型
#     ppo_agent.save("ppo_model.pth")


#加载训练模型
if __name__ == "__main__":
    # 环境选择和初始化
    maze = [
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 3],
    ]
    # maze = [
    #     [2, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 3],
    # ]

    # env = gym.make(env_name)
    env = MazeEnv(maze,n=10, mode="single",render_mode="show")

    state = env.reset()

    # 获取环境状态维度和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("state_dim = ",state_dim)
    print("action_dim = ",action_dim)

    # 初始化PPO
    ppo_agent = PPO(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, has_continuous_action_space=False)
    ppo_agent.load('ppo_model.pth')

    done=False
    while not done:
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        pygame.time.wait(100)  # 控制每步的显示时间
    pygame.quit()
