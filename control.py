import gymnasium as gym
import numpy as np
from scipy.linalg import solve_continuous_are, inv

env = gym.make('InvertedPendulum-v4')

g = 9.8 
mass = 1.0
length = 1.0
inertia = mass * length ** 2

A = np.array([[0, 1], [g/length, 0]])
B = np.array([[0], [1/inertia]])



def lqr(state, Q=np.eye(2), R=1.0):
    """
    Линейно-квадратичный регулятор (LQR) для управления перевёрнутым маятником.

    Параметры:
    state (numpy.ndarray): текущее состояние маятника [позиция, угол, линейная скорость, угловая скорость]
    Q (numpy.ndarray): матрица весов состояния, по умолчанию единичная матрица
    R (float): весовой коэффициент управляющего воздействия

    Возвращает:
    numpy.ndarray: оптимальное управляющее воздействие
    """
    angle, angular_velocity = state[1], state[3]
    state_vector = np.array([angle, angular_velocity])

    P = solve_continuous_are(A, B, Q, R)
    K = inv(R) @ B.T @ P
    u = -K @ state_vector
    return np.clip(u, -3.0, 3.0)

def pid(state):
    p = 1
    i = 0
    d = 1
    aim=np.pi/2

    ang = state[1]
    ang_vel=state[3]

    err = ang-aim
    if ang>aim and ang_vel>0:
        vel_err = -1*ang_vel
    else: vel_err = ang_vel
    
    return p*err+d*vel_err
    
