import numpy as np
import pickle

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def sat(x: float, limit: float) -> float:
    return clamp(x, -limit, limit)

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a*a)))

def sat_fraction(tau_L_cmd: np.ndarray, tau_R_cmd: np.ndarray, tau_max: float) -> float:
    satL = np.abs(tau_L_cmd) >= (tau_max - 1e-9)
    satR = np.abs(tau_R_cmd) >= (tau_max - 1e-9)
    return float(np.mean(np.logical_or(satL, satR)))

def save_results(filename: str, data: dict):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename: str) -> dict:
    with open(filename, "rb") as f:
        return pickle.load(f)