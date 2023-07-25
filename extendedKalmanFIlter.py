import numpy as np
from sympy import sin, cos, atan2, sqrt, Matrix
from sympy import symbols


class extendedKalmanFilter:
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

    def getJacobianA(self):
        return np.eye(3)

    def getJacobianB(self):
        u1, u2 = symbols("u1, u2")

        F = Matrix(
            [
                u1 * self.robot.dt * cos(self.state[2]),
                u1 * self.robot.dt * sin(self.state[2]),
                u2 * self.robot.dt,
            ]
        )
        Y = Matrix([u1, u2])

        return np.array(
            F.jacobian(Y).subs(
                [
                    (u1, self.vels[0]),
                    (u2, self.vels[1]),
                ]
            )
        ).astype(np.float64)

    def getJacobianC(self, L):
        self.L = L

        x1 = self.L[0][0]
        x2 = self.L[1][0]
        x3 = self.L[2][0]

        y1 = self.L[0][1]
        y2 = self.L[1][1]
        y3 = self.L[2][1]

        (xx, yy, tt) = symbols("xx, yy, tt")

        F = Matrix(
            [
                sqrt((x1 - xx) ** 2 + (y1 - yy) ** 2),
                sqrt((x2 - xx) ** 2 + (y2 - yy) ** 2),
                sqrt((x3 - xx) ** 2 + (y3 - yy) ** 2),
                atan2((y1 - yy), (x1 - xx) - tt),
                atan2((y2 - yy), (x2 - xx) - tt),
                atan2((y3 - yy), (x3 - xx) - tt),
            ]
        )

        Y = Matrix([xx, yy, tt])

        return np.array(
            F.jacobian(Y).subs(
                [
                    (xx, self.state[0]),
                    (yy, self.state[1]),
                    (tt, self.state[1]),
                ]
            )
        ).astype(np.float64)

    def predict(self, L):
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array(
            [
                [self.robot.dt * np.cos(self.state[2]), 0],
                [self.robot.dt * np.sin(self.state[2]), 0],
                [0, self.robot.dt],
            ]
        )
        self.C = np.array([[1, 0, 0], [0, 1, 0]])

        self.JA = self.getJacobianA()
        self.JB = self.getJacobianB()
        self.JC = self.getJacobianC(L)

        self.vels = np.array([self.robot.vl, self.robot.vr])
        self.x_hat = self.A.dot(self.state) + self.B.dot(self.vels)
        self.S_hat = self.JA.dot(self.S).dot(self.JA.T) + self.R
        return self.x_hat[:2]

    def correct(self):
        self.K = self.S_hat.dot(self.JB.T).dot(
            np.linalg.inv(self.JB.dot(self.S_hat).dot(self.JB.T) + self.Q)
        )
        self.state = self.x_hat + self.K.dot(self.meas - self.C.dot(self.x_hat))
        self.S = np.eye(3) - self.K.dot(self.C).dot(self.S_hat)
        return self.state[:2]
