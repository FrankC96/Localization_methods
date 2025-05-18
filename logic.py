import numpy as np
import pygame as pg

from environment import Environment
from robot import Robot


class Logic:
    def __init__(self, env: Environment, rbt: Robot):
        self.env = env
        self.robot = rbt

        x, y, _ = self.robot.state_vec

        # if robot collides with any Rect obstacle, remove it
        for obs in self.env.obstacles:
            if self._check_collision(obs.py_rect):
                # print("Removed obstacle")
                self.env.obstacles.remove(obs)

    def _check_collision(self, rect: pg.Rect, w: int=None, h: int=None):
        robot_x, robot_y, _ = self.robot.state_vec
        r = self.robot.radius

        if w:
            if (robot_x + r) > w:
                return True
            elif (robot_x - r) < 0:
                return True
            elif (robot_y + r) > h:
                return True
            elif (robot_y - r) < 0:
                return True
            
        for obs in self.env.obstacles:
            result = obs.py_rect.clipline(robot_x, robot_y, *obs.py_rect.center)
            
            if result:
                ln_start, _ = result
                dist = (robot_x - ln_start[0])**2 + (robot_y - ln_start[1])**2

                if dist - 2*r**2 < 0:
                    return True

    def check_for_collisions(self, screen: pg.display):
        width, height = screen.get_size()

        for obs in self.env.obstacles:
            if self._check_collision(obs.py_rect, width, height):
                self.robot.rewards -= 0.2
                self.robot.input_vec *= 0