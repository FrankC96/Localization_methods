import pygame
import numpy as np
from env.environment import *
from env.robot import Robot
from filters.kalmanFilter import kalmanFilter


def gameLoop():
    pygame.init()
    fps = 60
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

    robot = Robot(x0, (153, 204, 255), 1 / fps)
    ghost = Robot(x0, (160, 160, 160), 1 / fps)

    kf = kalmanFilter(robot, S, Q, R, x0, u, meas)

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
                    robot.vr += 0.1 * fps - 10
                    robot.moveRobot(x0)
                if event.key == pygame.K_DOWN:
                    robot.vl += 0
                    robot.vr -= 0.1 * fps - 10
                    robot.moveRobot(x0)

        screen.fill((255, 255, 255))

        robot.moveRobot(x0)
        x0 = [robot.x, robot.y, robot.theta]

        L = env.checkInCircle(robot)
        kf.predict()

        if not (None in L):
            xx, yy = robot.getOutput(L)
            kf.meas = np.array([xx, yy])
            kf.correct()

        ghost.x = kf.state[0]
        ghost.y = kf.state[1]
        ghost.theta = kf.state[2]

        env.draw(screen, robot)
        robot.drawRobot(screen)
        ghost.drawRobot(screen)

        pygame.display.flip()
        clock.tick(fps)


pygame.quit()
