import sympy as sp
import numpy as np
import numpy.typing as npt
import pygame as pg
from typing import List, ClassVar
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
    
    def measure(self, screen: pg.display, rx: int, ry: int):
        for obs in self.env.obstacles:
            result = obs.py_rect.clipline(rx, ry, self.radar_x, self.radar_y)

            if result:
                ln_start, ln_end = result

                pg.draw.circle(screen, (150, 255, 0), ln_start, 10)

                dist = float((rx - ln_start[0])**2 + (ry - ln_start[1])**2)
                return dist
            # else:
            #     return 999


        
class Radar:
    def __init__(self, env: Environment, length:int, n_sensors: int):
        self.env = env
        self.length = length
        self.n_sensors = n_sensors
        
        offsets = np.deg2rad(np.arange(0, 360, 360/self.n_sensors))
        self.sensors = [Sensor(self.env, off, self.length) for off in offsets]
    
    def draw(self, screen: pg.display, x: npt.NDArray[np.float64]):
        for sensor in self.sensors:
            r_x, r_y = sensor.extend(x)
            pg.draw.line(screen, (0, 180, 0), x[:2], (r_x, r_y), 2)

@dataclass
class Robot:
    _state_vec: npt.ArrayLike
    _input_vec: npt.ArrayLike

    radius: float
    color:  List
    dt:     float

    def __post_init__(self):    
        self.rewards = 0
        self._state_vec = self._state_vec.copy()
        self._input_vec = self._input_vec.copy()

        self.hist_reward = []
        self.hist_state = [self._state_vec.copy()] 
        self.hist_input = [self._input_vec.copy()] 
    
    # Setters / Getters for state and input
    @property
    def state_vec(self):
        return self._state_vec.copy()
    
    @property
    def input_vec(self):
        return self._input_vec.copy()
    
    @state_vec.setter
    def state_vec(self, x: npt.ArrayLike):
        self._state_vec = x.copy()  # Make a copy of new state
    
    @input_vec.setter
    def input_vec(self, u: npt.ArrayLike):
        #? If I am setting an input, I always need to move
        self._input_vec = u.copy()  # Make a copy of new input
        self._state_vec = self.move()

    def dynamics(self, x_st: npt.ArrayLike, u:npt.ArrayLike) -> npt.ArrayLike:
        """A mathematical model representing the robot's dynamics"""
        x, y, theta = x_st
        v, w = u.copy()

        if np.isclose(w, 0, 1E-4):
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
    
    def draw(self, screen: pg.display, trail: bool = False):
        """Draw the position of the robot on the screen"""
        x, y, theta = self._state_vec

        #! Calculate the robot's heading to a xy point
        r_x = x + np.cos(theta) * self.radius
        r_y = y + np.sin(theta) * self.radius

        pg.draw.circle(screen, self.color, (x, y), self.radius)
        pg.draw.line(screen, [255, 255, 255], (x, y), (r_x, r_y))

        if trail:
            for prev_state in self.hist_state:
                pg.draw.circle(screen, [0, 0, 0], prev_state[:2], 2)