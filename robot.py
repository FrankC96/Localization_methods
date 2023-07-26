import pygame
import numpy as np


class Robot:
    def __init__(self, x0: list, color: tuple, dt: float):
        self.color = color

        self.dt = dt
        self.vl = 0
        self.vr = 0

        self.x = x0[0]
        self.y = x0[1]
        self.theta = x0[2]

        self.r = []
        self.phi = []

        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array(
            [
                [self.dt * np.cos(self.theta), 0],
                [self.dt * np.sin(self.theta), 0],
                [0, self.dt],
            ]
        )

    def moveRobot(self, x):
        """
        Nonlinear dynamical model for a differential drive robot.
        It is formulated in matrix form, however it is not linear
        (although we use the linear kalman filter).
        """
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array(
            [
                [self.dt * np.cos(self.theta), 0],
                [self.dt * np.sin(self.theta), 0],
                [0, self.dt],
            ]
        )
        u = np.array([self.vl, self.vr])
        x = self.A.dot(x) + self.B.dot(u)

        self.x = x[0]
        self.y = x[1]
        self.theta = x[2]

    def drawRobot(self, screen):
        """
        Function to draw the robot's location and bearing.
        """
        r_x = self.x + np.cos(self.theta) * 40
        r_y = self.y + np.sin(self.theta) * 40

        pygame.draw.circle(screen, self.color, (self.x, self.y), 40)
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (r_x, r_y), 2)

    def getOutput(self, L):
        """
        Calculate the distances [r] and bearings [phi] for every
        landmark. Use distances [r] and positions [x, y] of every
        landmark in order to estimate the location of the robot.
        """
        self.r = []
        self.phi = []
        for i in range(len(L)):
            if L[i]:
                self.r.append(
                    np.linalg.norm(np.array([self.x, self.y]) - np.array(L[i]))
                )
                self.phi.append(
                    np.arctan2(
                        (np.array(L[i][1]) - self.y), (np.array(L[i][0] - self.x))
                    )
                    - self.theta
                )
            else:
                self.r.append(200)
                self.phi.append(None)
        # Square the distances
        d1_sq = self.r[0] ** 2
        d2_sq = self.r[1] ** 2
        d3_sq = self.r[2] ** 2

        x1 = L[0][0]
        x2 = L[1][0]
        x3 = L[2][0]

        y1 = L[0][1]
        y2 = L[1][1]
        y3 = L[2][1]

        # Formulate the equations
        A = 2 * (x2 - x1)
        B = 2 * (y2 - y1)
        C = d1_sq - d2_sq - x1**2 - y1**2 + x2**2 + y2**2

        D = 2 * (x3 - x1)
        E = 2 * (y3 - y1)
        F = d1_sq - d3_sq - x1**2 - y1**2 + x3**2 + y3**2

        # Solve the system of equations
        x = (C * E - F * B) / (E * A - B * D) + np.random.uniform(-10, 10)
        y = (C * D - A * F) / (B * D - A * E) + np.random.uniform(-10, 10)

        return x, y
