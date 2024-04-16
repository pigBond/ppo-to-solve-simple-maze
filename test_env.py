import pygame
import numpy as np
from env.maze_env import MazeEnv

# 创建迷宫和环境进行测试，可以通过修改 mode='path' 来切换显示模式
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
 [1, 1, 1, 1, 0, 1, 0, 0, 0, 3]
]


env = MazeEnv(maze, mode='path')
env.reset()
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    action = np.random.randint(0, 4)  # 随机选择一个动作
    state, reward, done = env.step(action)
    env.render()
    pygame.time.wait(500)  # 暂停500毫秒

pygame.quit()
