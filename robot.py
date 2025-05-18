import numpy as np
import numpy.typing as npt
import pygame as pg
from typing import List, Optional
from dataclasses import dataclass

from environment import Environment


class Sensor:
    def __init__(self, env: Environment, offset: float, length: int):
        self.env = env
        self.offset = offset
        self.length = length

    def extend(self, x: npt.NDArray[np.float64]):
        self.radar_x = x[0] + self.length * np.cos(x[2] + self.offset)
        self.radar_y = x[1] + self.length * np.sin(x[2] + self.offset)

        return self.radar_x, self.radar_y

    def measure(self, screen: pg.Surface, rx: int, ry: int) -> Optional[float]:
        for obs in self.env.obstacles:
            result = obs.py_rect.clipline(rx, ry, self.radar_x, self.radar_y)

            if result:
                ln_start, _ = result

                pg.draw.circle(screen, (150, 255, 0), ln_start, 10)

                dist = float((rx - ln_start[0]) ** 2 + (ry - ln_start[1]) ** 2)
                return dist


class Radar:
    def __init__(self, env: Environment, length: int, n_sensors: int):
        self.env = env
        self.length = length
        self.n_sensors = n_sensors

        offsets = np.deg2rad(np.arange(0, 360, 360 / self.n_sensors))
        self.sensors = [Sensor(self.env, off, self.length) for off in offsets]

    def draw(self, screen: pg.Surface, x: npt.NDArray[np.float64]):
        rob_x, rob_y, _ = x
        for sensor in self.sensors:
            r_x, r_y = sensor.extend(x)
            pg.draw.line(screen, (0, 180, 0), (rob_x, rob_y), (r_x, r_y), 2)


@dataclass
class Robot:
    _state_vec: npt.NDArray[np.float64]
    _input_vec: npt.NDArray[np.float64]

    radius: float
    color: List[int]
    dt: float

    def __post_init__(self):
        self.rewards: float = 0
        self._state_vec = self._state_vec
        self._input_vec = self._input_vec

        self.hist_reward: List[List[int]] = []
        self.hist_state = [self._state_vec]
        self.hist_input = [self._input_vec]

    # Setters / Getters for state and input
    @property
    def state_vec(self) -> npt.NDArray[np.float64]:
        return self._state_vec

    @property
    def input_vec(self):
        return self._input_vec

    @state_vec.setter
    def state_vec(self, x: npt.NDArray[np.float64]):
        self._state_vec = x

    @input_vec.setter
    def input_vec(self, u: npt.NDArray[np.float64]):
        # ? If I am setting an input, I always need to move
        self._input_vec = u
        self._state_vec = self.move()

    def dynamics(
        self, x_st: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """A mathematical model representing the robot's dynamics"""
        x, y, theta = x_st
        v, w = u.copy()

        if np.isclose(w, 0, 1e-4):
            w += 0.0001

        p1 = v / w

        # Calculate new state without modifying input
        new_x = x - p1 * np.sin(theta) + p1 * np.sin(theta + w * self.dt)
        new_y = y + p1 * np.cos(theta) - p1 * np.cos(theta + w * self.dt)
        new_theta = theta + w * self.dt

        return np.array([new_x, new_y, new_theta])

    def move(self):
        """Use dynamics to move one step forward"""
        self.hist_state.append(self._state_vec)
        self.hist_input.append(self._input_vec)
        self._state_vec = self.dynamics(self._state_vec, self._input_vec)
        return self._state_vec

    def draw(self, screen: pg.Surface, trail: bool = False):
        """Draw the position of the robot on the screen"""
        x, y, theta = self._state_vec

        #! Calculate the robot's heading to a xy point
        r_x = x + np.cos(theta) * self.radius
        r_y = y + np.sin(theta) * self.radius

        pg.draw.circle(screen, self.color, (x, y), self.radius)
        pg.draw.line(screen, [255, 255, 255], (x, y), (r_x, r_y))

        if trail:
            for prev_state in self.hist_state:
                x_prev, y_prev = prev_state[:2]
                pg.draw.circle(screen, [0, 0, 0], (x_prev, y_prev), 2)
