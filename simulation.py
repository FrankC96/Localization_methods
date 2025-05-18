import os
import time
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt

from robot import *
from controller import *
from utils import *
from kalmanFilter import kalmanFilter
from environment import Environment
from mlp import MLP


#! Check if robot stepped over a speck
def check_rewards(robot: Robot, env: Environment):
    x, y, _ = robot.state_vec
    r = robot.radius
    x_rew = env.targets[:, 0]
    y_rew = env.targets[:, 1]

    in_points = np.where((x - x_rew)**2 + (y - y_rew)**2 < r**2)
    for idx in in_points[0]:
        point = (float(x_rew[idx]), float(y_rew[idx]))
        if point not in robot.hist_reward:
            robot.rewards += 1
            robot.hist_reward.append(point)
    return robot.hist_reward

def game_loop(n_sim: int, controller: MLP):
    logger = Logger("sim_logs.csv")
    
    logger.init()
    pg.init()
    
    #! Pygame parameters
    FPS = 60
    WIDTH, HEIGHT = (1024, 768)
    screen = pg.display.set_mode([WIDTH, HEIGHT])
    clock = pg.time.Clock()

    #! Robot parameters
    xcurr = np.array([WIDTH // 2, HEIGHT//2, 0])
    ucurr = np.array([5, 0.1])
    xref = np.array([800, 100, np.deg2rad(180)])
    r1_col = [0, 0, 0]

    #! Robot creation
    DT = 0.001
    r1 = Robot(xcurr, ucurr, radius=30.0, color=r1_col, dt=DT)
    r_target = Robot(xref, ucurr, radius=30.0, color=[255, 0, 0], dt=DT)
    logger.log("IMPORTANT", f"Created object Robot r1")

    #! Environment creation
    env = Environment(500, (WIDTH, HEIGHT))

    #! Main loop
    for epoch in range(n_sim):
        for event in pg.event.get():
            if event.type == pg.QUIT:  
                running = False
        
        # screen.fill([255, 255, 255])
        
        r1.input_vec = controller.forward_pass(r1.state_vec)
        print(f"Got {r1.rewards} rewards")

        # CALCULATE REWARDS
        collected_specks = check_rewards(r1, env)

        # r1.draw(screen, trail=True)

        # env.draw_specks(screen, collected_specks)
        # pg.display.flip()

        clock.tick(FPS)

    logger.close()
    pg.quit()
    return r1.rewards
