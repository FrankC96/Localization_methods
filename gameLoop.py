import pygame
import numpy as np
from environment import *
from robot import Robot
from kalmanFilter import kalmanFilter
from extendedKalmanFIlter import extendedKalmanFilter
import matplotlib.pyplot as plt


def gameLoop():
    pygame.init()
    fps = 30
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode([800, 800])

    obs1 = (300, 500, 200, 200)
    obs2 = (500, 50, 200, 200)
    obs3 = (50, 200, 200, 200)

    obstacles = [obs1, obs2, obs3]

    bcs1 = (250, 250)
    bcs2 = (500, 250)
    bcs3 = (390, 500)

    beacons = [bcs1, bcs2, bcs3]

    env = Obstacles(obstacles, beacons)

    x0 = [350, 350, 0]
    S = np.eye(3)
    Q = 10 * np.eye(2)
    R = np.eye(3)

    u = np.array([0, 0])
    meas = np.array([0, 0])

    robot = Robot(x0, 1 / fps)

    kf = kalmanFilter(robot, S, Q, R, x0, u, meas)
    extKf = extendedKalmanFilter(robot, S, Q, R, x0, u, meas)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    robot.vl += 1 * fps
                    robot.vr += 0
                    robot.moveRobot(x0)
                if event.key == pygame.K_x:
                    robot.vl = 0
                    robot.vr = 0
                    robot.moveRobot(x0)
                if event.key == pygame.K_UP:
                    robot.vl += 0
                    robot.vr += 0.1 * fps
                    robot.moveRobot(x0)
                if event.key == pygame.K_DOWN:
                    robot.vl += 0
                    robot.vr -= 0.1 * fps
                    robot.moveRobot(x0)

        screen.fill((255, 255, 255))

        robot.moveRobot(x0)
        x0 = [robot.x, robot.y, robot.theta]

        L = env.checkInCircle(robot)
        e_pos = extKf.predict(L)

        pos = kf.predict()
        kf.state[0] = pos[0]
        kf.state[1] = pos[1]
        if not (None in L):
            xx, yy = robot.getOutput(L)
            kf.meas = np.array([xx, yy])
            pos = kf.correct()
            kf.state[0] = pos[0]
            kf.state[1] = pos[1]

        env.draw(screen, robot)
        robot.drawRobot(screen)

        pygame.draw.circle(screen, (0, 0, 0), pos, 10)

        pygame.display.flip()
        clock.tick(fps)


pygame.quit()
