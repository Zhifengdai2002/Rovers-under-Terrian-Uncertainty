import numpy as np
from config import RoverParams, EnvParams
from environment import slope_profile_rad

def rover_dynamics(
    state: np.ndarray,
    tau_L: float,
    tau_R: float,
    params: RoverParams,
    env: EnvParams,
    c_rr: float
) -> np.ndarray:
    x, y, th, v, w = state
    alpha = slope_profile_rad(x, env)

    F_drive = (tau_L + tau_R) / params.r_w
    M_drive = (tau_R - tau_L) * (params.b / (2.0 * params.r_w))

    F_slope = params.m * params.g * np.sin(alpha)  
    F_rr = c_rr * params.m * params.g * np.cos(alpha) * np.sign(v)  

    dv = (F_drive - F_slope - F_rr - params.c_v * v) / params.J_v
    dw = (M_drive - params.c_w * w) / params.J_w

    dx = v * np.cos(th)
    dy = v * np.sin(th)
    dth = w

    return np.array([dx, dy, dth, dv, dw], dtype=float)

def rk4_step(f, state, u, dt, *args):
    tau_L, tau_R = u
    k1 = f(state, tau_L, tau_R, *args)
    k2 = f(state + 0.5*dt*k1, tau_L, tau_R, *args)
    k3 = f(state + 0.5*dt*k2, tau_L, tau_R, *args)
    k4 = f(state + dt*k3, tau_L, tau_R, *args)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)