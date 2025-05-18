import csv
import sympy as sp
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Logger:
    path: str

    def __post_init__(self):
        self.path = Path(self.path)
        self.logs = []
    
    def init(self):
        self.file = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["TYPE     ", 5*"\t", "MESSAGE"])
        self.writer.writerow([100*"="])

    def log(self, msg_type: str, msg: str):
        self.logs.append([msg_type, 5*"\t", msg])
        self.writer.writerow(self.logs[-1])

    def close(self):
        self.file.close()

def compute_jacobians(x_state: npt.ArrayLike, u_inp: npt.ArrayLike, dt: float):
    x, y, theta = sp.symbols('x y theta')
    v, w = sp.symbols('v w') 

    if w != 0:
        p1 = v / w
    
    x_next = x - p1 * sp.sin(theta) + p1 * sp.sin(theta + w * dt)
    y_next = y + p1 * sp.cos(theta) - p1 * sp.cos(theta + w * dt)
    theta_next = theta + w * dt

    f = sp.Matrix([x_next, y_next, theta_next])

    state = sp.Matrix([x, y, theta])  # state vector
    inp = sp.Matrix([v, w])

    A = f.jacobian(state)
    B = f.jacobian(inp)

    A_numeric = sp.lambdify((x, y, theta, v, w), A, modules='numpy')
    B_numeric = sp.lambdify((x, y, theta, v, w), B, modules='numpy')

    x_val, y_val, theta_val = x_state
    v_val, w_val = u_inp

    return A_numeric(x_val, y_val, theta_val, v_val, w_val), B_numeric(x_val, y_val, theta_val, v_val, w_val)