import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from config import RoverParams

def plot_episode(out: Dict[str, np.ndarray], title: str, params: RoverParams):
    t, X, U, debug = out["t"], out["X"], out["U"], out["debug"]
    x, y, v = X[:,0], X[:,1], X[:,3]
    xr, yr = debug["xr"], debug["yr"]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=20)

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(xr, yr, label="reference")
    ax1.plot(x, y, label="actual")
    ax1.set_title("Trajectory", fontsize=16)
    ax1.legend(fontsize=11)

    ax2 = fig.add_subplot(2,2,2)
    epos = np.sqrt((xr-x)**2 + (yr-y)**2)
    ax2.plot(t, epos, label="position error")
    ax2.set_title("Tracking Error", fontsize=16)
    ax2.legend(fontsize=11)

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(t, v, label="v")
    ax3.plot(t, debug.get("vr", np.ones_like(t)*np.nan), label="v_ref")
    ax3.set_title("Speed", fontsize=16)
    ax3.legend(fontsize=11)

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(t, U[:,0], label="tau_L")
    ax4.plot(t, U[:,1], label="tau_R")
    ax4.axhline(params.tau_max, linestyle="--")
    ax4.axhline(-params.tau_max, linestyle="--")
    ax4.set_title("Wheel Torques", fontsize=16)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_summary(metrics_base: Dict[str,float], metrics_opt: Dict[str,float]):
    small = ["rmse_pos_mean", "rmse_pos_p90", "rmse_th_mean", "rmse_th_p90", "sat_frac_mean"]
    base_vals = [metrics_base[k] for k in small]
    opt_vals  = [metrics_opt[k]  for k in small]

    x = np.arange(len(small))
    width = 0.35
    fig1 = plt.figure(figsize=(11, 4))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.bar(x - width/2, base_vals, width, label="baseline")
    ax1.bar(x + width/2, opt_vals,  width, label="optimized")
    ax1.set_xticks(x)
    ax1.set_xticklabels(small, rotation=20, ha="right")
    ax1.set_title("Tracking & Constraints Metrics", fontsize=18)
    ax1.legend()
    plt.tight_layout()
    plt.show()

def plot_terrain_height(out: Dict[str, np.ndarray], title: str):
    x = out["X"][:, 0]
    slope_rad = np.deg2rad(out["debug"]["slope_deg"])
    dzdx = np.tan(slope_rad)
    dx = np.diff(x, prepend=x[0])
    z = np.cumsum(dzdx * dx)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, z, linewidth=2)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_tracking_error_comparison(out_base: Dict[str, np.ndarray], out_opt: Dict[str, np.ndarray]):
    t_base, t_opt = out_base["t"], out_opt["t"]
    epos_base = np.sqrt((out_base["debug"]["xr"] - out_base["X"][:, 0])**2 + (out_base["debug"]["yr"] - out_base["X"][:, 1])**2)
    epos_opt = np.sqrt((out_opt["debug"]["xr"] - out_opt["X"][:, 0])**2 + (out_opt["debug"]["yr"] - out_opt["X"][:, 1])**2)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_base, epos_base, linewidth=2, label="Baseline")
    ax.plot(t_opt, epos_opt, linewidth=2, label="Optimized")
    ax.set_title("Tracking Error Comparison", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_optimization_history(history):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(history["iter"], history["obj"], linewidth=2)
    ax.set_title("Optimization Convergence", fontsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()