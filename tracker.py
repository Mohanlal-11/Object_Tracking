import numpy as np


class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt

        # State:
        # [x, y, vx, vy]
        self.x = np.array([
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ])

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix
        # We only measure x and y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # State covariance
        self.P = np.eye(4) * 1000

        # Process noise
        self.Q = np.eye(4) * 0.1

        # Measurement noise
        self.R = np.eye(2) * 10

    def predict(self):
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    def update(self, measurement):
        z = np.array(measurement, dtype=float).reshape(2, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
