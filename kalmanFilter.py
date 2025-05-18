import sympy as sp
import numpy as np
import numpy.typing as npt


def compute_jacobians(x_state: npt.ArrayLike, u_inp: npt.ArrayLike, dt: float):
    x, y, theta = sp.symbols('x y theta')
    v, w  = sp.symbols('v w') 

    p1 = v / w

    x_next = x - p1 * sp.sin(theta) + p1 * sp.sin(theta + w * dt)
    y_next = y + p1 * sp.cos(theta) - p1 * sp.cos(theta + w * dt)
    theta_next = theta +  w * dt

    f = sp.Matrix([x_next, y_next, theta_next])

    state = sp.Matrix([x, y, theta])
    inp = sp.Matrix([v, w])

    A = f.jacobian(state)

    A_numeric = sp.lambdify((x, y, theta, v, w), A, modules='numpy')
 
    x_val, y_val, theta_val = x_state
    v_val, w_val = u_inp

    return A_numeric(x_val, y_val, theta_val, v_val, w_val)

class kalmanFilter:
    def __init__(self, model, S, Q, R, x, u, meas):
        """
        Initialize Extended Kalman Filter
        
        Args:
            model: System dynamics function
            S: Initial state covariance (3x3)
            Q: Process noise covariance (3x3)
            R: Measurement noise covariance (2x2)
            x: Initial state estimate (3x1)
            u: Initial input (2x1)
            meas: Initial measurement (2x1)
        """
        self.model = model

        self.S = S  # State covariance (3x3)
        self.S_hat = np.copy(S)  # Predicted state covariance (3x3)
        self.Q = Q  # Process noise (3x3)
        self.R = R  # Measurement noise (2x2)

        self._state_vec = x  # Current state estimate (3x1)
        self._state_hat = x  # Predicted state (3x1)
        self._input_vec = u  # Current input (2x1)
        self._meas = meas  # Current measurement (2x1)

        self.hist_data = [self._state_vec]

    @property
    def state_vec(self):
        return self._state_vec
    
    @property
    def input_vec(self):
        return self._input_vec
    
    @property
    def meas(self):
        return self._meas
    
    @state_vec.setter
    def state_vec(self, val: npt.ArrayLike):
        self._state_vec = val        

    @input_vec.setter
    def input_vec(self, val: npt.ArrayLike):
        self._input_vec = val

    @meas.setter
    def meas(self, val: npt.ArrayLike):
        self._meas = val
            
    def update(self, dt: float):
        self.G = compute_jacobians(self._state_vec, self._input_vec, dt)
        
        self.H = np.array([[1, 0, 0], [0, 1, 0]])

        self._state_hat = self.model(self._state_vec, self._input_vec)
        
        self.S_hat = self.G @ self.S @ self.G.T + self.Q

        innovation_cov = self.H @ self.S_hat @ self.H.T + self.R
        self.K = self.S_hat @ self.H.T @ np.linalg.inv(innovation_cov)

        measurement_residual = self._meas - self._state_hat[:2]
        self._state_vec = self._state_hat + self.K @ measurement_residual

        self.S = (np.eye(3) - self.K @ self.H) @ self.S_hat

        self.hist_data.append(self._state_vec)
        return self._state_vec, self.S
    
    def get_hist_data(self):
        return self.hist_data