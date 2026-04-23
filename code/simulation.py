import numpy as np
from typing import Dict
from config import Gains, RoverParams, EnvParams, SimConfig
from controller import TrackingController
from dynamics import rover_dynamics, rk4_step
from environment import slope_profile_deg
from utils import wrap_to_pi, rmse, sat_fraction

def simulate_once(
    gains: Gains,
    params: RoverParams,
    env: EnvParams,
    cfg: SimConfig,
    seed: int = 0,
    randomize_env: bool = True
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    if randomize_env:
        c_rr = float(rng.uniform(env.c_rr_range[0], env.c_rr_range[1]))
        slope_scale = float(rng.uniform(0.8, 1.2))
    else:
        c_rr = env.c_rr_nom
        slope_scale = 1.0

    env_local = EnvParams(
        c_rr_nom=env.c_rr_nom,
        c_rr_range=env.c_rr_range,
        slope_deg_max=env.slope_deg_max * slope_scale
    )

    state = np.array([0.0, 0.6, np.deg2rad(5.0), 0.0, 0.0], dtype=float)
    ctrl = TrackingController(gains, params)
    ctrl.reset()

    N = int(cfg.T / cfg.dt)
    ts = np.linspace(0, cfg.T, N)

    X = np.zeros((N, 5))
    U = np.zeros((N, 2))
    U_cmd = np.zeros((N, 2))
    
    debug = {
        "xr": np.zeros(N), "yr": np.zeros(N), "thr": np.zeros(N), "vr": np.zeros(N),
        "e_y": np.zeros(N), "e_th": np.zeros(N), "e_v": np.zeros(N),
        "w_ref": np.zeros(N), "slope_deg": np.zeros(N),
    }

    for k, t in enumerate(ts):
        tau_L, tau_R, dbg = ctrl.compute(t, state, cfg)
        U[k, :] = [tau_L, tau_R]
        U_cmd[k, :] = [dbg["tau_L_cmd"], dbg["tau_R_cmd"]]

        for key in ["xr", "yr", "thr", "vr", "e_y", "e_th", "e_v", "w_ref"]:
            debug[key][k] = dbg[key]
        debug["slope_deg"][k] = slope_profile_deg(state[0], env_local)

        X[k, :] = state
        state = rk4_step(rover_dynamics, state, (tau_L, tau_R), cfg.dt, params, env_local, c_rr)
        state[2] = wrap_to_pi(state[2])

    x = X[:, 0]; y = X[:, 1]; th = X[:, 2]
    epos = np.sqrt((debug["xr"] - x)**2 + (debug["yr"] - y)**2)
    sat_frac = sat_fraction(U[:,0], U[:,1], params.tau_max)

    return {
        "t": ts, "X": X, "U": U, "U_cmd": U_cmd, "debug": debug,
        "c_rr": np.array([c_rr]), "slope_scale": np.array([slope_scale]),
        "rmse_pos": np.array([rmse(epos)]),
        "rmse_heading": np.array([rmse(wrap_to_pi(debug["thr"] - th))]),
        "sat_frac": np.array([sat_frac]),
    }

def evaluate_gains(
    gains: Gains,
    params: RoverParams,
    env: EnvParams,
    cfg: SimConfig,
    n_trials: int = 8,
    seed0: int = 0
) -> Dict[str, float]:
    rmse_pos_list = []
    rmse_th_list = []
    sat_list = []

    for i in range(n_trials):
        out = simulate_once(gains, params, env, cfg, seed=seed0+i, randomize_env=True)
        rmse_pos_list.append(out["rmse_pos"][0])
        rmse_th_list.append(out["rmse_heading"][0])
        sat_list.append(out["sat_frac"][0])

    return {
        "rmse_pos_mean": float(np.mean(rmse_pos_list)),
        "rmse_pos_std": float(np.std(rmse_pos_list)),    
        "rmse_pos_p90": float(np.quantile(rmse_pos_list, 0.90)), 
        "rmse_th_mean": float(np.mean(rmse_th_list)),
        "rmse_th_p90": float(np.quantile(rmse_th_list, 0.90)),
        "sat_frac_mean": float(np.mean(sat_list)),
    }