import math
import numpy as np
from typing import Tuple
from config import SimConfig, EnvParams
from utils import wrap_to_pi

def reference_trajectory(t: float, cfg: SimConfig) -> Tuple[float, float, float, float]:
    v = cfg.v_ref
    if cfg.path_type == "line":
        return v * t, 0.0, 0.0, v
    if cfg.path_type == "circle":
        R = 10.0
        w = v / R
        x = R * np.sin(w * t)
        y = R * (1 - np.cos(w * t))
        th = wrap_to_pi(w * t)
        return float(x), float(y), float(th), v

    A, L = 1.5, 20.0
    x = v * t
    y = A * np.sin(2 * np.pi * x / L)
    dy_dx = A * (2 * np.pi / L) * np.cos(2 * np.pi * x / L)
    th = math.atan2(dy_dx, 1.0)
    return float(x), float(y), float(th), v

def reference_curvature(x_ref: float, cfg: SimConfig) -> float:
    if cfg.path_type in ["line"]: return 0.0
    if cfg.path_type == "circle": return 1.0 / 10.0

    A, L = 1.5, 20.0
    k = 2.0 * np.pi / L
    y1 = A * k * np.cos(k * x_ref)          
    y2 = -A * (k**2) * np.sin(k * x_ref)   
    kappa = y2 / ((1.0 + y1*y1) ** 1.5)
    return float(kappa)

def slope_profile_deg(x: float, env: EnvParams) -> float:
    a = env.slope_deg_max
    return float(a * (0.6*np.sin(2*np.pi*x/25.0) + 0.4*np.sin(2*np.pi*x/10.0)))

def slope_profile_rad(x: float, env: EnvParams) -> float:
    return np.deg2rad(slope_profile_deg(x, env))