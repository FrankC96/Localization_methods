import pygame as pg
import numpy as np

from typing import List


class Rewards:
    def __init__(self, x_bord: int, y_bord: int):
        self.x = np.random.randint(0, x_bord)
        self.y = np.random.randint(0, y_bord)

        self.point = [self.x, self.y]

    def draw(self, screen: pg.Surface, color: List[int]):
        pg.draw.circle(screen, color, (self.x, self.y), 5)


class Obstacle:
    def __init__(self, x_bord: int, y_bord: int):
        self.w = np.random.randint(50, 200)
        self.h = np.random.randint(50, 200)

        x_bord -= self.w
        y_bord -= self.h
        self.x = np.random.randint(0, x_bord)
        self.y = np.random.randint(0, y_bord)

        self.py_rect: pg.Rect = pg.Rect(self.x, self.y, self.w, self.h)

    def draw(self, screen: pg.Surface):
        pg.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.w, self.h), 5)


class Environment:
    def __init__(self, n_targets: int, n_obstacles: int, borders: List[int]):

        # initialize n_targets rewards
        self.rewards = [Rewards(borders[0], borders[1]) for _ in range(n_targets)]

        # initialize n_obstacles obstacles
        self.obstacles = [Obstacle(borders[0], borders[1]) for _ in range(n_obstacles)]

    def draw_rewards(self, screen: pg.Surface, p_rews: List[List[int]]):
        for rew in self.rewards:
            if rew.point in p_rews:
                _color: List[int] = [0, 0, 0]
            else:
                _color = [255, 0, 0]
            rew.draw(screen, _color)

    def draw_obstacles(self, screen: pg.Surface):
        for obs in self.obstacles:
            obs.draw(screen)
