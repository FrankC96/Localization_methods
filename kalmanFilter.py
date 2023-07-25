import numpy as np


class kalmanFilter:
    def __init__(self, robot, S, Q, R, x, u, meas):
        self.S = S
        self.Q = Q
        self.R = R

        self.state = x
        self.pos = (0, 0)
        self.vels = u
        self.meas = meas

        self.x_hat = x.copy()
        self.S_hat = np.empty([3, 3])

        self.robot = robot

    def predict(self):
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array(
            [
                [self.robot.dt * np.cos(self.state[2]), 0],
                [self.robot.dt * np.sin(self.state[2]), 0],
                [0, self.robot.dt],
            ]
        )
        self.C = np.array([[1, 0, 0], [0, 1, 0]])

        self.vels = np.array([self.robot.vl, self.robot.vr])
        self.x_hat = self.A.dot(self.state) + self.B.dot(self.vels)
        self.S_hat = self.A.dot(self.S).dot(self.A.T) + self.R
        return self.x_hat[:2]

    def correct(self):
        self.K = self.S_hat.dot(self.C.T).dot(
            np.linalg.inv(self.C.dot(self.S_hat).dot(self.C.T) + self.Q)
        )
        self.state = self.x_hat + self.K.dot(self.meas - self.C.dot(self.x_hat))
        self.S = np.eye(3) - self.K.dot(self.C).dot(self.S_hat)
        return self.state[:2]
