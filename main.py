import os
import pickle
import numpy as np
import pygame as pg

# import matplotlib.pyplot as plt

from robot import *
from utils import *
from mlp import *
from environment import *
from logic import *


#! Check if robot stepped over a speck
def check_rewards(robot: Robot, env: Environment) -> List[List[int]]:
    x, y, _ = robot.state_vec
    r = robot.radius

    rews = env.rewards
    x_rew = np.array([r.x for r in rews])
    y_rew = np.array([r.y for r in rews])

    in_points = np.where((x - x_rew) ** 2 + (y - y_rew) ** 2 < r**2)
    for idx in in_points[0]:
        point = [int(x_rew[idx]), int(y_rew[idx])]

        if point not in robot.hist_reward:
            robot.rewards += (
                1 * idx / 100
            )  # multiplying by idx for later rewards to be more impactfull
            robot.hist_reward.append(point)
    return robot.hist_reward


def game_loop(nsim: int, controller: "MLP"):
    logger = Logger("sim_logs.csv")

    logger.init()
    pg.init()

    #! Pygame parameters
    FPS: int = 60
    WIDTH: int = 1024
    HEIGHT: int = 768
    BORDERS: List[int] = [WIDTH, HEIGHT]

    screen = pg.display.set_mode([WIDTH, HEIGHT])
    clock = pg.time.Clock()

    #! Robot parameters
    xcurr = np.array([WIDTH // 2, HEIGHT // 2, 0])
    ucurr = np.array([0, 0.001])
    r1_col = [0, 0, 0]

    #! Robot creation
    DT = 1 / FPS
    r1 = Robot(xcurr, ucurr, radius=30.0, color=r1_col, dt=DT)
    logger.log("IMPORTANT", f"Created object Robot r1")

    #! Environment creation
    env = Environment(500, 10, BORDERS)

    #! Logic creation
    logic = Logic(env, r1)

    #! Radar creation
    radar = Radar(env, 200, 12)

    font = pg.font.Font(None, 50)

    #! Main loop
    for _ in range(nsim):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_t:
                    r1.input_vec += np.array([1, 0]) * FPS
                elif event.key == pg.K_LEFT:
                    r1.input_vec += np.array([0, -1])
                elif event.key == pg.K_RIGHT:
                    r1.input_vec += np.array([0, 1])
                elif event.key == pg.K_x:
                    r1.input_vec = np.array([0, 0])

        screen.fill([255, 255, 255])
        vel_text = font.render(
            f"v: {r1.input_vec[0]} - w: {r1.input_vec[1]}", True, (0, 0, 0)
        )
        screen.blit(vel_text, (WIDTH - 300, 50))

        logic.check_for_collisions(screen)

        r1.move()
        r1.draw(screen, trail=True)

        radar.draw(screen, r1.state_vec)

        measurements: List[float] = []
        for sensor in radar.sensors:
            dist: Optional[float] = sensor.measure(
                screen, r1.state_vec[0], r1.state_vec[1]
            )
            if dist is None:
                dist = 9999

            measurements.append(dist)

        r1.input_vec = controller.forward_pass(np.array(measurements))
        prev_rewards = check_rewards(r1, env)
        env.draw_rewards(screen, prev_rewards)

        env.draw_obstacles(screen)

        pg.display.flip()
        clock.tick(FPS)

    logger.close()
    pg.quit()

    return r1.rewards


if __name__ == "__main__":
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    os.environ.pop("SDL_VIDEODRIVER", None)
    game_loop(5000, data)
