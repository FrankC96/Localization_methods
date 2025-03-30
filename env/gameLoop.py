import pygame
import numpy as np
from env.environment import *
from env.robot import Robot
from filters.kalmanFilter import kalmanFilter
from env.mpc import Controller

def model(x, u, dt=1/20):
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = np.array(
        [
            [dt * np.cos(x[2]), 0],
            [dt * np.sin(x[2]), 0],
            [0, dt],
        ]
    )

    return np.dot(A, x) + np.dot(B, u)

def gameLoop():
    pygame.init()
    fps = 20
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode([1024, 1024])

    obs1 = (300, 500, 200, 200)
    obs2 = (500, 50, 200, 200)
    obs3 = (50, 200, 200, 200)

    obstacles = [obs1, obs2, obs3]

    bcs1 = (250, 250)
    bcs2 = (500, 250)
    bcs3 = (390, 500)

    beacons = [bcs1, bcs2, bcs3]

    env = Obstacles(obstacles, beacons)

    curr_pose = np.array([250, 350, 0.0])

    # FIXME: 1/fps 
    robot = Robot(curr_pose, (153, 204, 255), 1/20)

    Q = 12/fps*np.array([
        [1.0 ,.0 ,.0],
        [.0, 1.0, 0.],
        [.0, .0, 1.0]])
    R = 1/fps*np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    mpc_dms = Controller(
        constr_method="COLL",
        model=model,
        n_states=3,
        n_inputs=2,
        n_pred=20,
        t_max=1,
        Q=Q,
        R=R,
        state_bounds=[],
        minimize_method="SLSQP",
        term_constr=False,
    )
    trail = []
    x_ref = np.zeros([3,])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    robot._u += np.array([1*fps, 0])
                    robot.moveRobot(curr_pose)
                if event.key == pygame.K_x:
                    robot._u = np.array([0.0, 0.0])
                    robot.moveRobot(curr_pose)
                if event.key == pygame.K_UP:
                    robot._u += np.array([0.0, 0.1*fps-10])
                    robot.moveRobot(curr_pose)
                if event.key == pygame.K_DOWN:
                    robot._u += np.array([0.0, -(0.1*fps-10)])
                    robot.moveRobot(curr_pose)

        screen.fill((255, 255, 255))
        trail.append(robot._x[:2])
        for point in trail:
            pygame.draw.circle(screen, (160, 160, 160), (point[0], point[1]), 2)
        res = mpc_dms.optimize(curr_pose/1024, x_ref/1024)
        if not res["flag"]:
            print("Optimization failed")
        robot._u = res["u"][0] * fps * 1024
        robot.moveRobot(curr_pose)
        curr_pose = robot._x
        print(curr_pose)

        robot.drawRobot(screen)

        pygame.display.flip()
        clock.tick(fps)


pygame.quit()
