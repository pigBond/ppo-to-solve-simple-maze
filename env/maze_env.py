import pygame
import numpy as np

directions = ["上", "下", "左", "右"]

class MazeEnv:
    def __init__(self, maze, mode='single'):
        self.maze = np.array(maze)
        # 查找起点和终点坐标
        self.start = tuple(np.argwhere(self.maze == 2)[0])
        self.target = tuple(np.argwhere(self.maze == 3)[0])
        self.current_position = self.start
        self.step_count = 0
        self.block_size = 40  # 设置每个格子的大小
        self.mode = mode  # 'single' or 'path'
        self.visited = set()  # 用于存储访问过的位置
        self.effective_movement=False # 用于存储本次动作是否能产生有效的移动
        self.init_screen()

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.maze.shape[1] * self.block_size, self.maze.shape[0] * self.block_size))
        pygame.display.set_caption("Maze Visualization")
        self.font = pygame.font.Font(None, 36)  # 字体初始化
        self.render_static()  # 首次渲染静态元素

    def render_static(self):
        """ 渲染不会变的部分，如墙和路径 """
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                rect = pygame.Rect(y * self.block_size, x * self.block_size, self.block_size, self.block_size)
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 墙体是黑色
                elif self.maze[x, y] == 2:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # 起点是蓝色
                elif self.maze[x, y] == 3:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # 终点是绿色
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # 路是白色

    def step(self, action):
        x, y = self.current_position
        # 预测新位置 同时检查新位置是否合法
        if action == 0 and x-1 > 0:  # 向上
            x -= 1
            self.effective_movement=True
        elif action == 1 and x+1 < self.maze.shape[0] - 1:  # 向下
            x += 1
            self.effective_movement=True
        elif action == 2 and y-1 > 0:  # 向左
            y -= 1
            self.effective_movement=True
        elif action == 3 and y+1 < self.maze.shape[1] - 1:  # 向右
            y += 1
            self.effective_movement=True
        else:
            self.effective_movement=False

        if self.maze[x, y] == 1:
            # 保持在原位置
            return self.current_position, -5, False
        elif (x, y) == self.target:
            return (x, y), 20, True
        else:
            self.current_position = (x, y)
            self.visited.add(self.current_position)
            self.step_count += 1
            if self.effective_movement is True:
                print("=====================================================================")
                print("本次动作有效")
                print("动作: ",directions[action])
                print("得到位置: ",self.current_position)
                print("步数: ",self.step_count)
                print("=====================================================================")
                self.effective_movement=False
            return self.current_position, -1, False
        
    def render(self):
            if self.mode == 'single':
                self.render_static()  # 重新渲染静态元素，覆盖旧路径
            # 绘制访问过的路径（如果在路径模式下）
            if self.mode == 'path':
                for pos in self.visited:
                    rect = pygame.Rect(pos[1] * self.block_size, pos[0] * self.block_size, self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, (128, 128, 128), rect)

            # 绘制当前位置和步数
            x, y = self.current_position
            rect = pygame.Rect(y * self.block_size, x * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, (128, 128, 128), rect)  # 当前位置用灰色标记
            text = self.font.render(str(self.step_count), True, (0, 0, 0))
            # self.font.render 方法用于创建包含文本的图像
            text_rect = text.get_rect(center=(y * self.block_size + self.block_size // 2,
                                            x * self.block_size + self.block_size // 2))
            # text.get_rect() 方法获取文本图像的矩形区域，用于定位。
            # center= 设置文本图像的中心点。计算方法是当前块的中心位置，确保文字居中于矩形。           
            self.screen.blit(text, text_rect)

            pygame.display.flip()  # 更新整个屏幕
            
    def reset(self):
        self.current_position = self.start
        self.step_count = 0
        self.visited = set([self.start])  # 重置访问位置记录
        self.render_static()  # 重新渲染静态元素
        return self.current_position