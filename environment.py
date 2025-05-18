from pygame import display, draw, Rect
import numpy as np
import numpy.typing as npt

from typing import List


class Rewards:
    def __init__(self, x_bord: int, y_bord: int):
        self.x = np.random.randint(0, x_bord)
        self.y = np.random.randint(0, y_bord)

        self.point = (self.x, self.y)

    def draw(self, screen: display, color: List[int]):
        draw.circle(screen, color, (self.x, self.y), 5)

class Obstacle:
    def __init__(self, x_bord: int, y_bord: int):
        self.w = np.random.uniform(50, 200)
        self.h = np.random.uniform(200, 50)

        self.x = np.random.randint(0, x_bord - self.w)
        self.y = np.random.randint(0, y_bord - self.h)

        self.py_rect = Rect(self.x, self.y, self.w, self.h)
    
    def draw(self, screen: display):
        draw.rect(screen, (0, 0, 0), (self.x, self.y, self.w, self.h), 5)

        
class Environment:
    def __init__(self, n_targets: int, n_obstacles: int, borders: List):

        # initialize n_targets rewards
        self.rewards = [Rewards(borders[0], borders[1]) for _ in range(n_targets)]

        # initialize n_obstacles obstacles
        self.obstacles = [Obstacle(borders[0], borders[1]) for _ in range(n_obstacles)]

    def draw_rewards(self, screen: display, p_rews: List[List[np.float64]]):
        for rew in self.rewards:
            if rew.point in p_rews:
                _color = (0, 0, 0) 
            else:
                _color = (255, 0 , 0)
            rew.draw(screen, _color)
    
    def draw_obstacles(self, screen: display):
        for obs in self.obstacles:
            obs.draw(screen)